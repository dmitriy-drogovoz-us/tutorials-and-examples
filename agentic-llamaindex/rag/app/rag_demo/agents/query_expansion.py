from typing import List
from llama_index.llms.ollama import Ollama
import logging

logger = logging.getLogger(__name__)

class QueryExpander:
    def __init__(self, llm: Ollama):
        self.llm = llm

    def expand(self, query: str) -> List[str]:
        prompt = (
            "You are a helpful assistant that generates 3-5 expanded search terms for "
            "a movie recommendation query. Make the expansions explicit and diverse.\n\n"
            f"Original query: \"{query}\"\n\n"
            "Return a comma-separated list of 3-5 expanded queries or keywords."
        )
        resp = self.llm.complete(prompt)
        expanded_str = resp.text.strip()
        
        expanded_list = [q.strip() for q in expanded_str.split(',') if q.strip()]
        logger.info(f"Expanded query '{query}' to: {expanded_list}")
        
        return expanded_list[:5]