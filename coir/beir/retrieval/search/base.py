from abc import ABC, abstractmethod
from typing import Dict

class BaseSearch(ABC):

    @abstractmethod
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        pass
# MY CODE
    @abstractmethod
    def lexical_search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        pass

    @abstractmethod
    def lexical_search_bm25(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        pass

    @abstractmethod
    def hibrid_search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        pass

    @abstractmethod
    def hibrid_search_with_bm25(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        pass