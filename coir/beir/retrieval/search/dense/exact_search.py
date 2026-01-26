from .. import BaseSearch
from .util import cos_sim, dot_score
import logging
import torch
from typing import Dict
import heapq
import bm25s
logger = logging.getLogger(__name__)

# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearch(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
    
  # MY CODE START

    def lexical_search_bm25(self,
                        corpus: Dict[str, Dict[str, str]],
                        queries: Dict[str, str],
                        top_k: int,
                        **kwargs) -> Dict[str, Dict[str, float]]:

        query_ids = list(queries.keys())
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)

        corpus_texts = [corpus[cid].get("text", "") for cid in corpus_ids]

        tokenized_corpus = bm25s.tokenize(corpus_texts, stopwords="en")

        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)

        results = {qid: {} for qid in query_ids}
        for qid, query_text in queries.items():

            tokenized_query = bm25s.tokenize(query_text, stopwords="en")

            if(len(corpus_texts) < top_k):
                top_k = len(corpus_texts)
            doc_indexes, scores = retriever.retrieve(tokenized_query, k=top_k)

            for idx, score in zip(doc_indexes[0], scores[0]):
                if corpus_ids[idx] != qid:
                    results[qid][corpus_ids[idx]] = float(score)

        return results

    def lexical_search(self, 
                       corpus: Dict[str, Dict[str, str]], 
                       queries: Dict[str, str], 
                       top_k: int,  
                       **kwargs) -> Dict[str, Dict[str, float]]:
        
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries] # запросы текстом
        
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        docs = [doc['text'] for doc in corpus] #документы текстом

        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for query_iter in range(len(queries)):
            query_id = query_ids[query_iter]
            q_set = set(queries[query_iter].split())
            for doc_itr in range(len(docs)):
                corpus_id = corpus_ids[doc_itr]
                d_set = set(docs[doc_itr].split())
                intersection = len(q_set.intersection(d_set))
                union = len(q_set.union(d_set))
                score = intersection / union if union > 0 else 0.0 # рассчёт score!!! (0 <= score <=1)
                if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score 

        return self.results       

  # MY CODE END

    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
                
        
        return self.results 

# MY CODE START

    def lexical_search_fh(self, 
                       corpus: Dict[str, Dict[str, str]], 
                       queries: Dict[str, str], 
                       top_k: int,  
                       **kwargs):
        
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries] # запросы текстом
        
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        docs = [doc['text'] for doc in corpus] #документы текстом

        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for query_iter in range(len(queries)):
            query_id = query_ids[query_iter]
            q_set = set(queries[query_iter].split())
            for doc_itr in range(len(docs)):
                corpus_id = corpus_ids[doc_itr]
                d_set = set(docs[doc_itr].split())
                intersection = len(q_set.intersection(d_set))
                union = len(q_set.union(d_set))
                score = intersection / union if union > 0 else 0.0 # рассчёт score!!! (0 <= score <=1)
                if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        return result_heaps       

    def search_fh(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs):
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))                
        
        return result_heaps

    def _reciprocal_rank_fusion(self, semantic_results, lexical_results, top_k, k: int = 60):
        """
        RRF: score = 1/(rank + k) где k обычно 60
        Чем выше ранг документа в обоих списках, тем выше итоговый скор
        """
        fused_results = {}
        
        for query_id in semantic_results.keys():
            semantic_scores = semantic_results[query_id]
            lexical_scores = lexical_results[query_id]
            
            semantic_ranked = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
            lexical_ranked = sorted(lexical_scores.items(), key=lambda x: x[1], reverse=True)
            
            semantic_ranks = {}
            lexical_ranks = {}
            
            for rank, (doc_id, _) in enumerate(semantic_ranked):
                semantic_ranks[doc_id] = rank + 1 
                
            for rank, (doc_id, _) in enumerate(lexical_ranked):
                lexical_ranks[doc_id] = rank + 1

            all_docs = set(semantic_scores.keys()) | set(lexical_scores.keys())
            rrf_scores = {}
            
            for doc_id in all_docs:
                semantic_rank = semantic_ranks.get(doc_id, len(semantic_ranks) + k + 1)
                lexical_rank = lexical_ranks.get(doc_id, len(lexical_ranks) + k + 1)
                
                rrf_score = 1.0 / (semantic_rank + k) + 1.0 / (lexical_rank + k)
                rrf_scores[doc_id] = rrf_score
            
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results[query_id] = dict(sorted_docs)
        
        return fused_results

    def _weighted_score_fusion(self, semantic_results, lexical_results, top_k, alpha: float = 0.5):
        """
        Взвешенное среднее нормализованных скорингов
        alpha: вес для семантического поиска (0-1)
        """
        fused_results = {}
        
        for query_id in semantic_results.keys():
            semantic_scores = semantic_results[query_id]
            lexical_scores = lexical_results[query_id]
            
            if semantic_scores:
                sem_max = max(semantic_scores.values())
                sem_min = min(semantic_scores.values())
                if sem_max > sem_min:
                    semantic_norm = {doc_id: (score - sem_min) / (sem_max - sem_min) 
                                for doc_id, score in semantic_scores.items()}
                else:
                    semantic_norm = {doc_id: 1.0 for doc_id in semantic_scores.keys()}
            else:
                semantic_norm = {}
                
            if lexical_scores:
                lex_max = max(lexical_scores.values())
                lex_min = min(lexical_scores.values())
                if lex_max > lex_min:
                    lexical_norm = {doc_id: (score - lex_min) / (lex_max - lex_min) 
                                for doc_id, score in lexical_scores.items()}
                else:
                    lexical_norm = {doc_id: 1.0 for doc_id in lexical_scores.keys()}
            else:
                lexical_norm = {}
            
            all_docs = set(semantic_norm.keys()) | set(lexical_norm.keys())
            fused_scores = {}
            
            for doc_id in all_docs:
                sem_score = semantic_norm.get(doc_id, 0.0)
                lex_score = lexical_norm.get(doc_id, 0.0)
                
                fused_score = alpha * sem_score + (1 - alpha) * lex_score
                fused_scores[doc_id] = fused_score
            
            sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results[query_id] = dict(sorted_docs)
        
        return fused_results

    def _score_interpolation(self, semantic_results, lexical_results, top_k, alpha: float = 0.5):
        """
        Линейная интерполяция скорингов
        """
        fused_results = {}
        
        for query_id in semantic_results.keys():
            semantic_scores = semantic_results[query_id]
            lexical_scores = lexical_results[query_id]
            
            all_docs = set(semantic_scores.keys()) | set(lexical_scores.keys())
            fused_scores = {}
            
            for doc_id in all_docs:
                sem_score = semantic_scores.get(doc_id, 0.0)
                lex_score = lexical_scores.get(doc_id, 0.0)
                
                # Интерполяция
                fused_score = alpha * sem_score + (1 - alpha) * lex_score
                fused_scores[doc_id] = fused_score
            
            sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results[query_id] = dict(sorted_docs)
        
        return fused_results

    def _combMNZ_fusion(self, semantic_results, lexical_results, top_k):
        """
        CombMNZ: score = (sum of scores) * log(number of lists containing doc)
        """
        fused_results = {}
        
        for query_id in semantic_results.keys():
            semantic_scores = semantic_results[query_id]
            lexical_scores = lexical_results[query_id]
            
            # Нормализация z-score
            all_sem_scores = list(semantic_scores.values())
            all_lex_scores = list(lexical_scores.values())
            
            if all_sem_scores:
                sem_mean = sum(all_sem_scores) / len(all_sem_scores)
                sem_std = (sum((s - sem_mean) ** 2 for s in all_sem_scores) / len(all_sem_scores)) ** 0.5
                if sem_std > 0:
                    semantic_norm = {doc_id: (score - sem_mean) / sem_std 
                                for doc_id, score in semantic_scores.items()}
                else:
                    semantic_norm = {doc_id: 0.0 for doc_id in semantic_scores.keys()}
            else:
                semantic_norm = {}
                
            if all_lex_scores:
                lex_mean = sum(all_lex_scores) / len(all_lex_scores)
                lex_std = (sum((s - lex_mean) ** 2 for s in all_lex_scores) / len(all_lex_scores)) ** 0.5
                if lex_std > 0:
                    lexical_norm = {doc_id: (score - lex_mean) / lex_std 
                                for doc_id, score in lexical_scores.items()}
                else:
                    lexical_norm = {doc_id: 0.0 for doc_id in lexical_scores.keys()}
            else:
                lexical_norm = {}
            
            # CombMNZ
            all_docs = set(semantic_norm.keys()) | set(lexical_norm.keys())
            combmnz_scores = {}
            
            for doc_id in all_docs:
                scores = []
                if doc_id in semantic_norm:
                    scores.append(semantic_norm[doc_id])
                if doc_id in lexical_norm:
                    scores.append(lexical_norm[doc_id])
                
                combmnz_score = sum(scores) * len(scores)  # sum * log(num_lists) но обычно просто num_lists
                combmnz_scores[doc_id] = combmnz_score
            
            sorted_docs = sorted(combmnz_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results[query_id] = dict(sorted_docs)
        
        return fused_results

    def hibrid_search_with_bm25(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        
        fusion_method = "combMNZ"
        alpha = 0.5

        if len(queries) < top_k:
            top_k = len(queries)
            print("\n\n\nNew top_k:")
            print(top_k)
            print("\n\n\n")

        semantic_result = self.search(corpus, queries, top_k, score_function, return_sorted, **kwargs)

        lexical_result = self.lexical_search_bm25(corpus, queries, top_k)

        # !!!DENCHIK!!! lexical_result rerank !!!DENCHIK!!!
        # vstavlay suda.
 
        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(semantic_result, lexical_result, top_k)
        elif fusion_method == "weighted":
            return self._weighted_score_fusion(semantic_result, lexical_result, top_k, alpha)
        elif fusion_method == "interpolation":
            return self._score_interpolation(semantic_result, lexical_result, top_k, alpha)
        elif fusion_method == "combMNZ":
            return self._combMNZ_fusion(semantic_result, lexical_result, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")


    def hibrid_search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:

        lexical_result = self.lexical_search_fh(corpus, queries, top_k)
        
        semantic_result = self.search_fh(corpus, queries, top_k, score_function, return_sorted, **kwargs)

        #merge

        for qid in lexical_result:
            heapq.heapify_max(lexical_result[qid])
            heapq.heapify_max(semantic_result[qid])
            if top_k > len(lexical_result[qid]):
                border = len(lexical_result[qid])
            else:
                border = top_k
            count = 0
            addedCorpusIds = []
            f1 = 0
            while count < border/2:
                score, corpus_id = heapq.heappop_max(semantic_result[qid])
                self.results[qid][corpus_id] = score
                addedCorpusIds.append(corpus_id)
                count+=1
                f1 += 1
            f2 = 0
            while count < border:
                score, corpus_id = heapq.heappop_max(lexical_result[qid])
                if(corpus_id not in addedCorpusIds):
                    self.results[qid][corpus_id] = score
                    count+=1
                    f2 += 1

            print(f1, f2)
        #for qid in lexical_result:           
        #    for score, corpus_id in semantic_result[qid]:
        #        if corpus_id in self.results[qid]:
        #            if self.results[qid][corpus_id] < score:
        #                self.results[qid][corpus_id] = score 
        #        else:
        #            self.results[qid][corpus_id] = score
        #
        #    for score, corpus_id in lexical_result[qid]:
        #        if corpus_id in self.results[qid]:
        #            if self.results[qid][corpus_id] < score:
        #                self.results[qid][corpus_id] = score 
        #        else:
        #            self.results[qid][corpus_id] = score

        return self.results

            


        


# MY CODE END