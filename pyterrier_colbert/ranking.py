
import os
import torch
import pandas as pd
import pyterrier as pt
assert pt.started(), "please run pt.init() before importing pyt_colbert"

from pyterrier import tqdm
from pyterrier.datasets import Dataset
from typing import Union, Tuple
from colbert.evaluation.load_model import load_model
from . import load_checkpoint
# monkeypatch to use our downloading version
import colbert.evaluation.loaders
colbert.evaluation.loaders.load_checkpoint = load_checkpoint
colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.modeling.inference import ModelInference
from colbert.evaluation.slow import slow_rerank
from colbert.indexing.loaders import get_parts, load_doclens
import colbert.modeling.colbert
from collections import defaultdict
import numpy as np
import pickle
from warnings import warn

class file_part_mmap:
    def __init__(self, file_path, file_doclens, dim):
        self.dim = dim
        
        self.doclens = file_doclens
        self.endpos = np.cumsum(self.doclens)
        self.startpos = self.endpos - self.doclens

        mmap_storage = torch.HalfStorage.from_file(file_path, False, sum(self.doclens) * self.dim)
        self.mmap = torch.HalfTensor(mmap_storage).view(sum(self.doclens), self.dim)
 
    def get_embedding(self, pid):
        startpos = self.startpos[pid]
        endpos = self.endpos[pid]
        return self.mmap[startpos:endpos,:]

class file_part_mem:
    def __init__(self, file_path, file_doclens, dim):
        self.dim = dim

        self.doclens = file_doclens
        self.endpos = np.cumsum(self.doclens)
        self.startpos = self.endpos - self.doclens

        self.mmap = torch.load(file_path)
        #print(self.mmap.shape)
 
    def get_embedding(self, pid):
        startpos = self.startpos[pid]
        endpos = self.endpos[pid]
        return self.mmap[startpos:endpos,:]


class Object(object):
    pass


from typing import List     


class re_ranker_mmap:
    def __init__(self, index_path, args, inference, verbose = False, memtype='mmap'):
        self.args = args
        self.doc_maxlen = args.doc_maxlen
        assert self.doc_maxlen > 0
        self.inference = inference
        self.dim = args.dim
        self.verbose = verbose
    
        # Every pt file gets its own list of doc lengths
        self.part_doclens = load_doclens(index_path, flatten=False)
        assert len(self.part_doclens) > 0, "Did not find any indices at %s" % index_path
        # Local mmapped tensors with local, single file accesses
        self.part_mmap : List[file_part_mmap] = re_ranker_mmap._load_parts(index_path, self.part_doclens, self.dim, memtype)
        
        # last pid (inclusive, e.g., the -1) in each pt file
        # the -1 is used in the np.searchsorted
        # so if each partition has 1000 docs, the array is [999, 1999, ...]
        # this helps us map from passage id to part (inclusive, explaning the -1)
        self.part_pid_end_offsets = np.cumsum([len(x) for x in self.part_doclens]) - 1

        self.segment_sizes = torch.LongTensor([0] + [x.mmap.shape[0] for x in self.part_mmap])
        self.segment_starts = torch.cumsum(self.segment_sizes, 0)
        
        # first pid (inclusive) in each pt file
        tmp = np.cumsum([len(x) for x in self.part_doclens])
        tmp[-1] = 0
        self.part_pid_begin_offsets = np.roll(tmp, 1)
        # [0, 1000, 2000, ...]
        self.part_pid_begin_offsets
    
    @staticmethod
    def _load_parts(index_path, part_doclens, dim, memtype="mmap"):
        # Every pt file is loaded and managed independently, with local pids
        _, all_parts_paths, _ = get_parts(index_path)
        
        if memtype == "mmap":
            all_parts_paths = [ file.replace(".pt", ".store") for file in all_parts_paths ]
            mmaps = [file_part_mmap(path, doclens, dim) for path, doclens in zip(all_parts_paths, part_doclens)]
        elif memtype == "mem":
            mmaps = [file_part_mem(path, doclens, dim) for path, doclens in tqdm(zip(all_parts_paths, part_doclens), total=len(all_parts_paths), desc="Loading index shards to memory", unit="shard")]
        else:
            assert False, "Unknown memtype %s" % memtype
        return mmaps

    def num_docs(self):
        """
        Return number of documents in the index
        """
        return sum([len(x) for x in self.part_doclens])

    def get_embedding(self, pid):
        # In which pt file we need to look the given pid
        part_id = np.searchsorted(self.part_pid_end_offsets, pid)
        # calculate the pid local to the correct pt file
        local_pid = pid - self.part_pid_begin_offsets[part_id]
        # identify the tensor we look for
        disk_tensor = self.part_mmap[part_id].get_embedding(local_pid)
        doclen = disk_tensor.shape[0]
         # only here is there a memory copy from the memory mapped file 
        target = torch.zeros(self.doc_maxlen, self.dim)
        target[:doclen, :] = disk_tensor
        return target
    
    def get_embedding_copy(self, pid, target, index):
        # In which pt file we need to look the given pid
        part_id = np.searchsorted(self.part_pid_end_offsets, pid)
        # calculate the pid local to the correct pt file
        local_pid = pid - self.part_pid_begin_offsets[part_id]
        # identify the tensor we look for
        disk_tensor = self.part_mmap[part_id].get_embedding(local_pid)
        doclen = disk_tensor.shape[0]
        # only here is there a memory copy from the memory mapped file 
        target[index, :doclen, :] = disk_tensor
        return target
    
    def our_rerank(self, query, pids, gpu=True):
        colbert = self.args.colbert
        inference = self.inference

        Q = inference.queryFromText([query])
        if self.verbose:
            pid_iter = tqdm(pids, desc="lookups", unit="d")
        else:
            pid_iter = pids

        D_ = torch.zeros(len(pids), self.doc_maxlen, self.dim)
        for offset, pid in enumerate(pid_iter):
            self.get_embedding_copy(pid, D_, offset)

        if gpu:
            D_ = D_.cuda()

        scores = colbert.score(Q, D_).cpu()
        del(D_)
        return scores.tolist()

    def our_rerank_batched(self, query, pids, gpu=True, batch_size=1000):
        import more_itertools
        if len(pids) < batch_size:
            return self.our_rerank(query, pids, gpu=gpu)
        allscores=[]
        for group in more_itertools.chunked(pids, batch_size):
            batch_scores = self.our_rerank(query, group, gpu)
            allscores.extend(batch_scores)
        return allscores
        
        
    def our_rerank_with_embeddings(self, qembs, pids, weightsQ=None, gpu=True):
        """
        input: qid,query, docid, query_tokens, query_embeddings, query_weights 
        
        output: qid, query, docid, score
        """
        colbert = self.args.colbert
        inference = self.inference
        # default is uniform weight for all query embeddings
        if weightsQ is None:
            weightsQ = torch.ones(len(qembs))
        # make to 3d tensor
        Q = torch.unsqueeze(qembs, 0)
        if gpu:
            Q = Q.cuda()
        
        if self.verbose:
            pid_iter = tqdm(pids, desc="lookups", unit="d")
        else:
            pid_iter = pids

        D_ = torch.zeros(len(pids), self.doc_maxlen, self.dim)
        for offset, pid in enumerate(pid_iter):
            self.get_embedding_copy(pid, D_, offset)
        if gpu:
            D_ = D_.cuda()
        maxscoreQ = (Q @ D_.permute(0, 2, 1)).max(2).values.cpu()
        scores = (weightsQ*maxscoreQ).sum(1).cpu()
        return scores.tolist()

    def our_rerank_with_embeddings_batched(self, qembs, pids, weightsQ=None, gpu=True, batch_size=1000):
        import more_itertools
        if len(pids) < batch_size:
            return self.our_rerank_with_embeddings(qembs, pids, weightsQ, gpu)
        allscores=[]
        for group in more_itertools.chunked(pids, batch_size):
            batch_scores = self.our_rerank_with_embeddings(qembs, group, weightsQ, gpu)
            allscores.extend(batch_scores)
        return allscores

class ColBERTModelOnlyFactory():

    def __init__(self, 
            colbert_model : Union[str, Tuple[colbert.modeling.colbert.ColBERT, dict]], gpu=True, mask_punctuation=False, dim=128):
        args = Object()
        args.query_maxlen = 32
        args.doc_maxlen = 180
        args.dim = dim  
        args.bsize = 128
        args.similarity = 'cosine'        
        args.amp = True
        args.nprobe = 10 #### THIS IS THE NUMBER OF CLUSTERS SEARCHED
        args.part_range = None
        args.mask_punctuation = mask_punctuation

        self.gpu = True
        if not gpu:
            
            warn("Gpu disabled, YMMV")
            import colbert.parameters
            import colbert.evaluation.load_model
            import colbert.modeling.colbert
            colbert.parameters.DEVICE = colbert.evaluation.load_model.DEVICE = colbert.modeling.colbert.DEVICE = torch.device("cpu")
            self.gpu = False
        if isinstance (colbert_model, str):
            args.checkpoint = colbert_model
            args.colbert, args.checkpoint = load_model(args)
        else:
            assert isinstance(colbert_model, tuple)
            args.colbert, args.checkpoint = colbert_model
            from colbert.modeling.colbert import ColBERT
            assert isinstance(args.colbert, ColBERT)
            assert isinstance(args.checkpoint, dict)
            
        args.inference = ModelInference(args.colbert, amp=args.amp)
        self.args = args
        
    def query_encoder(self, detach=True) -> pt.Transformer:
        """
        Returns a transformer that can encode queries using ColBERT's model.
        input: qid, query
        output: qid, query, query_embs, query_toks,
        """
        def _encode_query(row):
            with torch.no_grad():
                Q, ids, masks = self.args.inference.queryFromText([row.query], bsize=512, with_ids=True)
                if detach:
                    Q = Q.cpu()
                return pd.Series([Q[0], ids[0]])
            
        def row_apply(df):
            if "docno" in df.columns or "docid" in df.columns:
                warn("You are query encoding an R dataframe, the query will be encoded for each row")
            df[["query_embs", "query_toks"]] = df.apply(_encode_query, axis=1)
            return df
        
        return pt.apply.generic(row_apply)
    
    def text_encoder(self, detach=True, batch_size=8) -> pt.Transformer:
        """
        Returns a transformer that can encode the text using ColBERT's model.
        input: qid, text
        output: qid, text, doc_embs, doc_toks,
        """
        def chunker(seq, size):
            for pos in range(0, len(seq), size):
                yield seq.iloc[pos:pos + size]
        def df_apply(df):
            with torch.no_grad():
                rtr_embs = []
                rtr_toks = []
                for chunk in chunker(df, batch_size):
                    embsD, idsD = self.args.inference.docFromText(chunk.text.tolist(), with_ids=True)
                    if detach:
                        embsD = embsD.cpu()
                    rtr_embs.extend([embsD[i, : ,: ] for i in range(embsD.shape[0])])
                    rtr_toks.extend(idsD)
            df["doc_embs"] = pd.Series(rtr_embs)
            df["doc_toks"] = pd.Series(rtr_toks)
            return df
        return pt.apply.generic(df_apply)

    def explain_text(self, query : str, document : str):
        """
        Provides a diagram explaining the interaction between a query and the text of a document
        """
        embsD, idsD = self.args.inference.docFromText([document], with_ids=True)
        return self._explain(query, embsD, idsD)
    
    def _explain(self, query, embsD, idsD):
        embsQ, idsQ, masksQ = self.args.inference.queryFromText([query], with_ids=True)

        interaction = (embsQ[0] @ embsD[0].T).cpu().numpy().T
        
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        tokenmap = {"[unused1]" : "[D]", "[unused0]" : "[Q]"}

        fig = plt.figure(figsize=(4, 12)) 
        gs = GridSpec(2, 1, height_ratios=[1, 20]) 

        ax1=fig.add_subplot(gs[0])
        ax2=fig.add_subplot(gs[1])
        
        ax2.matshow(interaction, cmap=plt.cm.Blues)
        qtokens = self.args.inference.query_tokenizer.tok.convert_ids_to_tokens(idsQ[0])
        dtokens = self.args.inference.query_tokenizer.tok.convert_ids_to_tokens(idsD[0])
        qtokens = [tokenmap[t] if t in tokenmap else t for t in qtokens]
        dtokens = [tokenmap[t] if t in tokenmap else t for t in dtokens]

        ax2.set_xticks(range(32), minor=False)
        ax2.set_xticklabels(qtokens, rotation=90)
        ax2.set_yticks(range(len(idsD[0])))
        ax2.set_yticklabels(dtokens)
        ax2.set_anchor("N")

        contributions=[]
        for i in range(32):
            maxpos = np.argmax(interaction[:,i])
            plt.text(i-0.25, maxpos+0.1, "X", fontsize=5)
            contributions.append(interaction[maxpos,i])

        from sklearn.preprocessing import minmax_scale
        ax1.bar([0.5 + i for i in range(0,32)], contributions, color=plt.cm.Blues(minmax_scale(contributions, feature_range=(0.4, 1))))
        ax1.set_xlim([0,32])
        ax1.set_xticklabels([])
        fig.tight_layout()
        #fig.subplots_adjust(hspace=-0.37)
        return fig

    def text_scorer(self, query_encoded=False, doc_attr="text", verbose=False) -> pt.Transformer:
        """
        Returns a transformer that uses ColBERT model to score the *text* of documents.
        """
        #input: qid, query, docno, text
        #OR
        #input: qid, query, query_embs, query_toks, query_weights, docno, text

        #output: qid, query, docno, score

        def slow_rerank_with_qembs(args, qembs, pids, passages, gpu=True):
            inference = args.inference

            # make to 3d tensor
            Q = torch.unsqueeze(qembs, 0)
            if gpu:
                Q = Q.cuda()
            
            D_ = inference.docFromText(passages, bsize=args.bsize)
            if gpu:
                D_ = D_.cuda()
            
            scores = (Q @ D_.permute(0, 2, 1)).max(2).values.sum(1)

            scores = scores.sort(descending=True)
            ranked = scores.indices.tolist()

            ranked_scores = scores.values.tolist()
            ranked_pids = [pids[position] for position in ranked]
            ranked_passages = [passages[position] for position in ranked]

            return list(zip(ranked_scores, ranked_pids, ranked_passages))

        def _text_scorer(queries_and_docs):
            groupby = queries_and_docs.groupby("qid")
            rtr=[]
            with torch.no_grad():
                for qid, group in tqdm(groupby, total=len(groupby), unit="q") if verbose else groupby:
                    query = group["query"].values[0]
                    ranking = slow_rerank(self.args, query, group["docno"].values, group[doc_attr].values.tolist())
                    for rank, (score, pid, passage) in enumerate(ranking):
                            rtr.append([qid, query, pid, score, rank])          
            return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

        # when query is encoded 
        def _text_scorer_qembs(queries_and_docs):
            groupby = queries_and_docs.groupby("qid")
            rtr=[]
            with torch.no_grad():
                for qid, group in tqdm(groupby, total=len(groupby), unit="q") if verbose else groupby:
                    qembs = group["query_embs"].values[0]
                    query = group["query"].values[0]
                    ranking = slow_rerank_with_qembs(self.args, qembs, group["docno"].values, group[doc_attr].values.tolist(), gpu=self.gpu)
                    for rank, (score, pid, passage) in enumerate(ranking):
                            rtr.append([qid, query, pid, score, rank])          
            return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

        return pt.apply.generic(_text_scorer_qembs if query_encoded else _text_scorer)

    def scorer(factory, add_contributions=False, add_exact_match_contribution=False, verbose=False, gpu=True) -> pt.Transformer:
        """
        Calculates the ColBERT max_sim operator using previous encodings of queries and documents
        input: qid, query_embs, [query_weights], docno, doc_embs
        output: ditto + score, [+ contributions]
        """
        import torch
        import pyterrier as pt
        assert pt.started(), 'PyTerrier must be started'
        cuda0 = torch.device('cuda') if gpu else torch.device('cpu')

        def _build_interaction(row, D):
            """
            Builds the interaction matrix by copying document embeddings to the interaction matrix.

            Args:
                row (object): The row object containing document embeddings.
                D (ndarray): The interaction matrix.

            Returns:
                None
            """
            doc_embs = row.doc_embs
            doc_len = doc_embs.shape[0]
            D[row.row_index, 0:doc_len, :] = doc_embs
            
        def _build_toks(row, idsD):
            """
            Build tokens for a document and store them in the idsD array.

            Args:
                row (object): The row object containing the document tokens.
                idsD (ndarray): The array to store the document tokens.

            Returns:
                None
            """
            doc_toks = row.doc_toks
            doc_len = doc_toks.shape[0]
            idsD[row.row_index, 0:doc_len] = doc_toks
        
        def _score_query(df):
            """
            Scores the query embeddings against the document embeddings in the given DataFrame.

            Args:
                df (pandas.DataFrame): DataFrame containing query and document embeddings.

            Returns:
                pandas.DataFrame: DataFrame with scores and additional columns if specified.

            Scores similarity between a query and multiple documents.

            df contains multiple rows, each corresponding to a document. Each row contains 
            the embeddings for that document, as well as the embeddings for a query
            
            The query embeddings are retrieved with df.iloc[0].query_embs, 
            same query is used for all documents in the DataFrame

            The function first concatenates the query embeddings into a tensor Q. If the 
            DataFrame contains query weights, these are also retrieved; otherwise, a tensor 
            of ones is used as the weights. If a GPU is available, the query embeddings and 
            weights are moved to the GPU.

            Next, a 3D tensor D is created to hold the document embeddings. Each row of D 
            corresponds to a document and is populated with the document's embeddings 
            by the _build_interaction function.

            The function then computes the maximum score for each document by taking the 
            dot product of the query embeddings and the document embeddings, and taking 
            the maximum value over the second dimension. This results in a tensor of maximum 
            scores, one for each document.

            The final scores are computed by multiplying the maximum scores by the query 
            weights and summing over the first dimension. These scores are then added to 
            the DataFrame as a new column.

            If the add_contributions flag is set, the function also computes the contributions 
            of each token in the query to the score and adds these to the DataFrame as a new column.

            If the add_exact_match_contribution flag is set, the function also builds a 
            tensor of token ids for the documents and computes a mask indicating which 
            tokens in the query are not special tokens (such as MASK tokens). This could 
            be used later to compute the contribution of exact matches to the score.
            """
            with torch.no_grad():
                weightsQ = None
                Q = torch.cat([df.iloc[0].query_embs])
                if "query_weights" in df.columns:
                    weightsQ = df.iloc[0].query_weights
                else:
                    weightsQ = torch.ones(Q.shape[0])
                if gpu:
                    Q = Q.cuda()
                    weightsQ = weightsQ.cuda()
                
                D = torch.zeros(len(df), factory.args.doc_maxlen, factory.args.dim, device=cuda0)
                
                # D: (num_documents, doc_len, emb_dim)
                # D: (N, 180, 128)
                # N is equal to k'(the number of documents to retrieve in the first stage)
                # k' comes from the line `ann_retrieve_score % k1`

                df['row_index'] = range(len(df))
                if verbose:
                    pt.tqdm.pandas(desc='scorer')
                    df.progress_apply(lambda row: _build_interaction(row, D), axis=1)
                else:
                    df.apply(lambda row: _build_interaction(row, D), axis=1)
                
                # Q: (query_len, emb_dim)
                # Q: (32, 128)
                # D: (k', 180, 128)

                # Q @ D.permute(0,2,1) = (32, 128) x (k', 128, 180) = (k', 32, 180)
                maxscoreQ = (Q @ D.permute(0, 2, 1)).max(2).values # maxscoreQ = (N, 32, )
                # scores = maxscoreQ.sum(1) = (k', )
                scores = (weightsQ*maxscoreQ).sum(1).cpu()
                # print(scores.shape) this prints torch.Size([1000]) when k1 = 1000
                df["score"] = scores.tolist()
                if add_contributions:
                    contributions = (Q @ D.permute(0, 2, 1)).max(1).values.cpu()
                    df["contributions"] = contributions.tolist()
                if add_exact_match_contribution:
                    idsQ = torch.cat([df.iloc[0].query_toks]).unsqueeze(0)
                    idsD = torch.zeros(len(df), factory.args.doc_maxlen, dtype=idsQ.dtype)

                    df.apply(lambda row: _build_toks(row, idsD), axis=1)

                    # which places in the query are actual tokens, not specials such as MASKs
                    token_match = (idsQ != 101) & (idsQ != 102) & (idsQ != 103) & (idsQ != 1) & (idsQ != 2)

                    # which places in the interaction have exact matches (not [CLS])
                    exact_match = (idsQ.unsqueeze(1) == idsD.unsqueeze(2)) & (idsQ != 101)
                    
                    # perform the interaction
                    interaction = (Q @ D.permute(0, 2, 1)).cpu()

                    weightsQ = weightsQ.unsqueeze(0).cpu()

                    weighted_maxsim = weightsQ*interaction.max(2).values

                    # mask out query embeddings that arent tokens 
                    weighted_maxsim[:, ~token_match[0,:]] = 0
                    
                    # get the sum
                    denominator = weighted_maxsim.sum(1)

                    # zero out entries that arent exact matches
                    interaction[~ exact_match.permute(0, 2, 1)] = 0

                    weighted_maxsim = weightsQ*interaction.max(2).values
                    # mask out query embeddings that arent tokens 
                    weighted_maxsim[:, ~token_match[0,:]] = 0

                    # get the sum
                    numerator = weighted_maxsim.sum(1)
                    #df["exact_count"] = exact_match
                    df["exact_numer"] = numerator.tolist()
                    df["exact_denom"] = denominator.tolist()
                    df["exact_pct"] = (numerator/denominator).tolist()
            return df
            
        return pt.apply.by_query(_score_query, add_ranks=True)

class ColBERTFactory(ColBERTModelOnlyFactory):

    def __init__(self, 
            colbert_model : Union[str, Tuple[colbert.modeling.colbert.ColBERT, dict]], 
            index_root : str, 
            index_name : str,
            faiss_partitions=None,#TODO 100-
            memtype = "mem",
            faisstype= "mem",
            **kwargs):
        
        super().__init__(colbert_model, **kwargs)
       
        self.verbose = False
        self._faissnn = None
        self.index_root = index_root
        self.index_name = index_name
        self.numdocs = -1
        if index_root is None or index_name is None:
            warn("No index_root and index_name specified - no index ranking possible")
        else:
            self.index_path = os.path.join(index_root, index_name)
            docnos_file = os.path.join(self.index_path, "docnos.pkl.gz")
            if os.path.exists(docnos_file):
                with pt.io.autoopen(docnos_file, "rb") as f:
                    self.docid2docno = pickle.load(f)
                    # support reverse docno lookup in memory
                    self.docno2docid = { docno : docid for docid, docno in enumerate(self.docid2docno) }
                    self.docid_as_docno = False
                self.numdocs = len(self.docid2docno)
            else:
                self.docid_as_docno = True
        
        self.faiss_index_on_gpu = True
        if not self.gpu:
            self.faiss_index_on_gpu = False

        try:
            import faiss
        except:
            warn("Faiss not installed. You cannot do retrieval")
        
        self.args.partitions = faiss_partitions
        self.faisstype = faisstype
        self.memtype = memtype

        #we load this lazily
        self.rrm = None
        self.faiss_index = None
        
    # allows a colbert index to be built from a dataset
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',
            **kwargs):
        
        from pyterrier.batchretrieve import _from_dataset
        
        #colbertfactory doesnt match quite the expectations, so we can use a wrapper fb
        def _ColBERTFactoryconstruct(folder, **kwargs):
            import os
            index_loc = os.path.dirname(folder)
            index_name = os.path.basename(folder)
            checkpoint = kwargs.get('colbert_model')
            del(kwargs['colbert_model'])
            return ColBERTFactory(checkpoint, index_loc, index_name, **kwargs)
        
        return _from_dataset(dataset, 
                             variant=variant, 
                             version=version, 
                             clz=_ColBERTFactoryconstruct, **kwargs)
        
    def _rrm(self) -> re_ranker_mmap:
        """
        Returns an instance of the re_ranker_mmap class.
        Only one is created, if necessary.
        """

        if self.rrm is not None:
            return self.rrm
        print("Loading reranking index, memtype=%s" % self.memtype)
        self.rrm = re_ranker_mmap(
            self.index_path, 
            self.args, 
            self.args.inference, 
            verbose=self.verbose, 
            memtype=self.memtype)
        return self.rrm

    def load_index(self, embeddings=True, faiss=False):
        """
        Forces the embeddigs and faiss indices to be loaded (into memory).
        If memtype or faisstype are set to mmap, then the mmapping is performed instead.
        """
        if embeddings:
            self._rrm()
        if faiss:
            self._faiss_index()

    def __len__(self):
        if self.numdocs > -1:
            return self.numdocs
        return self._rrm().num_docs()
        
    def nn_term(self, cf=True, df=False):
        """
        Returns an instance of the FaissNNTerm class, which provides statistics about terms
        """
        if self._faissnn is not None:
            return self._faissnn
        from .faiss_term_index import FaissNNTerm
        #TODO accept self.args.inference as well
        self._faissnn = FaissNNTerm(
            self.args.colbert,
            self.index_root,
            self.index_name,
            faiss_index = self._faiss_index(), 
            cf=cf, df=df, mask_punctuation=self.args.mask_punctuation)
        return self._faissnn

    def _faiss_index(self):
        """
        Returns an instance of the Colbert FaissIndex class, which provides nearest neighbour information
        """
        import glob
        from colbert.indexing.faiss import get_faiss_index_name
        from colbert.ranking.faiss_index import FaissIndex
        if self.faiss_index is not None:
            return self.faiss_index
        faiss_index_name = get_faiss_index_name(self.args)  # This should return something like 'ivfpq.faiss'
        index_name = faiss_index_name.split('.')[0]
        faiss_index_files = glob.glob(os.path.join(self.index_path, f"{index_name}*.faiss"))
        if not faiss_index_files:
            raise ValueError(f"No FAISS index found matching pattern {index_name}*.faiss in {self.index_path}")
        faiss_index_path = faiss_index_files[0] # choose the first index file that matches the pattern
        if len(faiss_index_files) > 1:
            warn(f"More than one FAISS index file matching the pattern {index_name}*.faiss found in {self.index_path}\nChoosing the first one which is at {faiss_index_path}")
        self.faiss_index = FaissIndex(self.index_path, faiss_index_path, self.args.nprobe, self.args.part_range, mmap=self.faisstype == 'mmap')
        # ensure the faiss_index is transferred to GPU memory for speed
        import faiss
        if self.faiss_index_on_gpu:
            self.faiss_index.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index.faiss_index)
        return self.faiss_index

    def set_retrieve(self, batch=False, query_encoded=False, faiss_depth=1000, verbose=False, docnos=False) -> pt.Transformer:
        """
        Performs ANN retrieval, but the retrieval forms a set - i.e. there is no score attribute. Number of documents retrieved
        is indirectly controlled by the faiss_depth parameters (denoted as k' in the original ColBERT paper).
        """
        #input: qid, query
        #OR
        #input: qid, query, query_embs, query_toks, query_weights

        #output: qid, query, docid, [docno]
        #OR
        #output: qid, query, query_embs, query_toks, query_weights, docid, [docno]
        
        assert not batch
        faiss_index = self._faiss_index()
        
        # this is when queries have NOT already been encoded
        def _single_retrieve(queries_df):
            # we know that query_encoded=False
            if "query_embs" in queries_df.columns:
                warn("set_retrieve() used with query_encoded=False, but query_embs column present in input. Should you use query_encoded=True?")
            rtr = []
            iter = queries_df.itertuples()
            iter = tqdm(iter, unit="q")  if verbose else iter
            for row in iter:
                qid = row.qid
                query = row.query
                with torch.no_grad():
                    Q, ids, masks = self.args.inference.queryFromText([query], bsize=512, with_ids=True)
                Q_f = Q[0:1, :, :]
                all_pids = faiss_index.retrieve(faiss_depth, Q_f, verbose=verbose)
                Q_cpu = Q[0, :, :].cpu()
                for passage_ids in all_pids:
                    if verbose:
                        print("qid %s retrieved docs %d" % (qid, len(passage_ids)))
                    for pid in passage_ids:
                        rtr.append([qid, query, pid, ids[0], Q_cpu])
                        
            #build the DF to return for this query
            rtrDf = pd.DataFrame(rtr, columns=["qid","query",'docid','query_toks','query_embs'] )
            if docnos:
                rtrDf = self._add_docnos(rtrDf)
            return rtrDf

        # this is when queries have already been encoded
        def _single_retrieve_qembs(queries_df):

            rtr = []
            query_weights = "query_weights" in queries_df.columns
            iter = queries_df.itertuples()
            iter = tqdm(iter, unit="q") if verbose else iter
            for row in iter:
                qid = row.qid
                embs = row.query_embs
                Q_f = torch.unsqueeze(embs, 0)
                all_pids = faiss_index.retrieve(faiss_depth, Q_f, verbose=verbose)
                for passage_ids in all_pids:
                    if verbose:
                        print("qid %s retrieved docs %d" % (qid, len(passage_ids)))
                    for pid in passage_ids:
                        if query_weights:
                           rtr.append([qid, row.query, pid, row.query_toks, row.query_embs, row.query_weights])
                        else:
                           rtr.append([qid, row.query, pid, row.query_toks, row.query_embs])
            
            #build the DF to return for this query
            cols = ["qid","query",'docid','query_toks','query_embs']
            if query_weights:
                cols.append("query_weights")
            rtrDf = pd.DataFrame(rtr, columns=cols)
            if docnos:
                rtrDf = self._add_docnos(rtrDf)
            return rtrDf
        
        return pt.apply.generic(_single_retrieve_qembs if query_encoded else _single_retrieve)

    def _add_docids(self, df):
        if self.docid_as_docno:
            df["docid"] = df["docno"].astype('int64')
        else:
            df["docid"] = df["docno"].apply(lambda docno : self.docno2docid[docno])
        return df

    def _add_docnos(self, df):
        if self.docid_as_docno:
            df["docno"] = df["docid"].astype('str')
        else:
            df["docno"] = df["docid"].apply(lambda docid : self.docid2docno[docid])
        return df

    def index_scorer(self, query_encoded=False, add_ranks=False, add_docnos=True, batch_size=10000) -> pt.Transformer:
        """
        Returns a transformer that uses the ColBERT index to perform scoring of documents to queries 
        """
        #input: qid, query, [docno], [docid] 
        #OR
        #input: qid, query, query_embs, query_toks, query_weights, docno], [docid] 

        #output: qid, query, docno, score

        rrm = self._rrm()

        def rrm_scorer(qid_group):
            if "query_embs" in qid_group.columns:
                warn("index_scorer() used with query_encoded=False, but query_embs column present in input. Should you use query_encoded=True?")
            qid_group = qid_group.copy()
            if "docid" not in qid_group.columns:
                qid_group = self._add_docids(qid_group)
            qid_group.sort_values("docid", inplace=True)
            docids = qid_group["docid"].values
            if batch_size > 0:
                scores = rrm.our_rerank_batched(qid_group.iloc[0]["query"], docids, batch_size=batch_size, gpu=self.gpu)
            else:
                scores = rrm.our_rerank(qid_group.iloc[0]["query"], docids, gpu=self.gpu)
            qid_group["score"] = scores
            if "docno" not in qid_group.columns and add_docnos:
                qid_group = self._add_docnos(qid_group)
            if add_ranks:
                return pt.model.add_ranks(qid_group)
            return qid_group

        def rrm_scorer_query_embs(qid_group):
            qid_group = qid_group.copy()
            if "docid" not in qid_group.columns:
                qid_group = self._add_docids(qid_group)
            qid_group.sort_values("docid", inplace=True)
            docids = qid_group["docid"].values
            weights = None
            if "query_weights" in qid_group.columns:
                weights = qid_group.iloc[0].query_weights
            if batch_size > 0:
                scores = rrm.our_rerank_with_embeddings_batched(qid_group.iloc[0]["query_embs"], docids, weights, batch_size=batch_size, gpu=self.gpu)
            else:
                scores = rrm.our_rerank_with_embeddings(qid_group.iloc[0]["query_embs"], docids, weights, gpu=self.gpu)
            qid_group["score"] = scores
            if "docno" not in qid_group.columns and add_docnos:
                qid_group = self._add_docnos(qid_group)
            if add_ranks:
                return pt.model.add_ranks(qid_group)
            return qid_group

        if query_encoded:
            return pt.apply.by_query(rrm_scorer_query_embs) 
        return pt.apply.by_query(rrm_scorer) 

    def get_embeddings_by_token(self, tokenid, flatten=True, sample=None) -> Union[torch.TensorType, List[torch.TensorType]]:
        """
        Returns all embeddings for a given tokenid. Specifying a sample fraction results in the embeddings being sampled.
        """
        import torch
        from typing import List

        def partition(tensor : torch.Tensor, offsets : torch.Tensor) -> List[torch.Tensor]:
            num_shards = offsets.shape[0]
            positions = torch.bucketize(tensor, offsets[1:])
            rtr = [tensor[positions == shard_id] for shard_id in range(num_shards)]
            return rtr
        
        nn_term = self.nn_term()
        rrm = self._rrm()
        
        # global positions of this tokenid in the entire index
        offsets = (nn_term.emb2tid == tokenid).nonzero()
        # apply sampling if requested
        if sample is not None:
            offsets = offsets[self.rng.integers(0, len(offsets), int(sample * len(offsets)))]
        # get offsets partitioned by index shard
        partitioned_offsets = partition(offsets, self.segment_starts)            
        # get the requested embeddings
        all_tensors = [ rrm.part_mmap[shard].mmap[shard_portion - self.segment_starts[shard]] for shard, shard_portion in enumerate(partitioned_offsets) if shard_portion.shape[0] > 0 ]
        # if requested, make a single tensor - involves a further copy
        if flatten:
            all_tensors = torch.cat(all_tensors).squeeze()
        return all_tensors

    def end_to_end(self) -> pt.Transformer:
        """

        ------------ VANILLA COLBERT ----------------

        Returns a transformer composition that uses a ColBERT FAISS index to retrieve documents, followed by a ColBERT index 
        to perform accurate scoring of the retrieved documents. Equivalent to `colbertfactory.set_retrieve() >> colbertfactory.index_scorer()`.
        """
        #input: qid, query, 
        #output: qid, query, docno, score
        return self.set_retrieve() >> self.index_scorer(query_encoded=True)

    def ann_retrieve_score(
            self,
            batch=False,
            query_encoded=False,
            faiss_depth=1000,
            verbose=False,
            maxsim=True,
            add_ranks=True,
            add_docnos=True,
            num_qembs_hint=32,
            num_docs_to_shuffle=0,
            verbose_=False
        ) -> pt.Transformer:
        """
        ------------ UPDATED COLBERT(from the paper below) ----------------
        Like set_retrieve(), uses the ColBERT FAISS index to retrieve documents, but scores them using the maxsim on the approximate (quantised) nearest neighbour scores. 

        Parameters:
        - batch(bool): whether to process all queries at once. True not currently supported.
        - query_encoded(bool): whether to apply the ColBERT model to encode the queries. Defaults to false.
        - faiss_depth(int): How many passage embeddings to retrieve for each query embedding, denoted as k' in the ColBERT paper. Defaults to 1000, as per the ColBERT paper.
        - verbose(bool): Display tqdm progress bar during retrieval
        - maxsim(bool): Whether to use approx maxsim (True) or approx sumsim (False). See our CIKM 2021 paper for more details. Default is True.
        - add_ranks(bool): Whether to use add the rank column, to allow rank cutoffs to be applied. Default is True. Response time will be enhanced if False.
        - add_docnos(bool):  Whether to use add the docno column. Default is True. Response time will be enhanced if False.

        Reference:
        C. Macdonald, N. Tonellotto. On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval
        In Proceedings of ICTIR CIKM.
        """
        assert not batch, "batching not supported yet"
        assert hasattr(self._faiss_index(), 'faiss_index'), "multi index support removed"
        assert maxsim, "only maxsim supported now."

        # this is a big malloc, sized for the number of docs in the collection
        # for this reason, we reuse it across queries. All used values are reset
        # to zero afer use.
        import numpy as np
        score_buffer = np.zeros( (len(self), num_qembs_hint ) )

        def _single_retrieve(queries_df):
            rtr = []
            weights_set = "query_weights" in queries_df.columns
            iter = queries_df.itertuples()
            iter = tqdm(iter, unit="q") if verbose else iter
            for row in iter:
                qid = row.qid
                if query_encoded:
                    embs = row.query_embs
                    qtoks = row.query_toks
                    ids = np.expand_dims(qtoks, axis=0)
                    Q_cpu = embs.cpu()
                    Q_cpu_numpy = embs.float().numpy()
                    #NB: ids is 2D
                    qweights = row.query_weights.unsqueeze(0) if weights_set else torch.ones(ids.shape)
                else:
                    if "query_embs" in queries_df.columns:
                        warn("ann_retrieve_score() used with query_encoded=False, but query_embs column present in input. Should you use query_encoded=True?")
                    with torch.no_grad():
                        Q, ids, masks = self.args.inference.queryFromText([row.query], bsize=512, with_ids=True)
                    Q_f = Q[0:1, :, : ]
                    Q_cpu = Q[0, :, :].cpu()
                    Q_cpu_numpy = Q_cpu.float().numpy()
                    #NB: ids is 2D
                    qweights = torch.ones(ids.shape)
                
                # retrieve faiss_depth=1000 most similar doc embeddings
                all_scores, all_embedding_ids = self._faiss_index().faiss_index.search(Q_cpu_numpy, faiss_depth)

                nonlocal score_buffer
                # we reuse this buffer, but it has to be big enough for expanded queries
                if score_buffer.shape[1] < Q_cpu_numpy.shape[0]:
                  score_buffer = np.zeros( (len(self), Q_cpu_numpy.shape[0] ) )
                
                pids, final_scores = _approx_maxsim_numpy(
                    all_scores, 
                    all_embedding_ids, 
                    self.faiss_index.emb2pid.numpy(), 
                    qweights[0].numpy(), 
                    score_buffer, 
                    num_docs_to_shuffle,
                    verbose_
                )

                for offset in range(pids.shape[0]):
                    rtr.append([qid, row.query, pids[offset], final_scores[offset], ids[0], Q_cpu, qweights[0]])

            rtr = pd.DataFrame(rtr, columns=["qid","query",'docid', 'score','query_toks','query_embs', 'query_weights'])
            if add_docnos:
                rtr = self._add_docnos(rtr)
            if add_ranks:
                rtr = pt.model.add_ranks(rtr)
            return rtr
        t = pt.apply.by_query(_single_retrieve, add_ranks=False, verbose=verbose)
        import types
        def __reduce_ex__(t2, proto):
            kwargs = { 'batch':batch, 'query_encoded': query_encoded, 'faiss_depth' : faiss_depth, 'maxsim': maxsim}
            return (
                ann_retrieve_score,
                #self is the factory, and it will be serialised using its own __reduce_ex__ method
                (self, [], kwargs),
                None
            )
        t.__reduce_ex__ = types.MethodType(__reduce_ex__, t)
        t.__getstate__ = types.MethodType(lambda t2 : None, t)
        return t

    def fetch_index_encodings(factory, verbose=False, ids=False) -> pt.Transformer:
        """
        New encoder that gets embeddings from rrm and stores into doc_embs column.
        If ids is True, then an additional doc_toks column is also added. This requires 
        a Faiss NN term index data structure, i.e. indexing should have ids=True set.
        input: docid, ...
        output: ditto + doc_embs [+ doc_toks]
        """
        def _get_embs(df):
            rrm = factory._rrm() # _rrm() instead of rrm because we need to check it has already been loaded.
            if verbose:
                import pyterrier as pt
                pt.tqdm.pandas(desc="fetch_index_encodings")
                df["doc_embs"] = df.docid.progress_apply(rrm.get_embedding) 
            else:
                df["doc_embs"] = df.docid.apply(rrm.get_embedding)
            return df

        def _get_tok_ids(df):
            fnt = factory.nn_term(False)
            def _get_toks(pid):
                end = fnt.end_offsets[pid]
                start = end - fnt.doclens[pid]
                return fnt.emb2tid[start:end].clone()

            if verbose:
                import pyterrier as pt
                pt.tqdm.pandas()
                df["doc_toks"] = df.docid.progress_apply(_get_toks)
            else:
                df["doc_toks"] = df.docid.apply(_get_toks)
            return df
        rtr = pt.apply.by_query(_get_embs, add_ranks=False)
        if ids:
            rtr = rtr >> pt.apply.by_query(_get_tok_ids, add_ranks=False)
        return rtr

    def explain_doc(self, query : str, doc : Union[str,int]):
        """
        Provides a diagram explaining the interaction between a query and a given docno
        """
        if isinstance(doc,str):
            pid = self.docno2docid[doc]
        elif isinstance(doc,int):
            pid = doc
        else:
            raise ValueError("Expected docno(str) or docid(int)")
        embsD = self._rrm().get_embedding(pid)
        idsD = self.nn_term().get_tokens_for_doc(pid)
        return self._explain(query, embsD, idsD)

import pandas as pd

def _approx_maxsim_numpy(faiss_scores, faiss_ids, mapping, weights, score_buffer, verbose=False, num_docs_to_shuffle=0):
    """
    Compute the approximate maximum similarity scores for each query-document pair using numpy.

    Args:
        faiss_scores: a 2D numpy array where each row corresponds to a query and each column corresponds to a document. The entries in this array are the similarity scores between the queries and documents as computed by the FAISS index.
        
        faiss_ids: a 2D numpy array with the same shape as faiss_scores. The entries in this array are the ids of the documents in the FAISS index.
        
        mapping: a 1D numpy array that maps the ids in the FAISS index to the ids in the original document collection.
        
        weights: a 1D numpy array that contains the weights for each query.
        
        score_buffer: a 2D numpy array that is used to store the maximum similarity scores. The rows in this array correspond to the documents and the columns correspond to the queries.

    Returns:
        tuple: A tuple containing the unique document IDs and the final similarity scores.
    """
    import numpy as np
    # faiss_depth: the number of documents returned by the FAISS index for each query
    faiss_depth = faiss_scores.shape[1]
    # pids: mapping the document ids returned by the FAISS index to the ids in the original document collection
    pids = mapping[faiss_ids]
    # qemb_ids: ids of the queries
    qemb_ids = np.arange(faiss_ids.shape[0])
    """
    iterate over the ranks of the documents returned by the FAISS index
    
    for each rank, update the score_buffer to contain the maximum similarity 
    score for each query and document pair
    """
    for rank in range(faiss_depth):
        rank_pids = pids[:, rank]
        score_buffer[rank_pids, qemb_ids] = np.maximum(score_buffer[rank_pids, qemb_ids], faiss_scores[:, rank])
    
    all_pids = np.unique(pids) # ids of the processed documents
    if verbose:
        print("all_pids.shape: ", all_pids.shape) # (N,) where N is number of unique documents retrieved at first stage
    """
    compute the final scores by summing the maximum similarity scores 
    for each document, weighted by the query weights
    """
    final = np.sum(score_buffer[all_pids, : ] * weights, axis=1)
    if verbose:
        print("final.shape: ", final.shape) # (N,) - same shape as all_pids
    # reset the score_buffer for the processed documents to zero
    score_buffer[all_pids, : ] = 0

    """
    The below part shuffles passage ids and their corresponding scores
    before returning them. Hopefully, this correctly investigates the
    effect of document orderings on retrieval effectiveness(especially for small
    datasets such as Vaswani).
    # Combine `all_pids` and `final` into a single list of tuples
    combined = list(zip(all_pids, final))

    # Shuffle the combined lists
    np.random.shuffle(combined)

    # Separate the shuffled structure back into `all_pids` and `final`
    all_pids, final = zip(*combined)
    # Convert them back to numpy arrays(might be redundant)
    all_pids = np.array(all_pids)
    final = np.array(final)
    """
    if num_docs_to_shuffle is not 0:
        pids_to_shuffle = all_pids[:num_docs_to_shuffle, :]
        final_scores_to_shuffle = final[:num_docs_to_shuffle, :]
        # Combine `all_pids` and `final` into a single list of tuples
        combined = list(zip(pids_to_shuffle, final_scores_to_shuffle))

        # Shuffle the combined lists
        np.random.shuffle(combined)

        # Separate the shuffled structure back into `all_pids` and `final`
        all_pids[:num_docs_to_shuffle, :], final[:num_docs_to_shuffle, :] = zip(*combined)
        # Convert them back to numpy arrays(might be redundant)
        all_pids = np.array(all_pids)
        final = np.array(final)

    return all_pids, final