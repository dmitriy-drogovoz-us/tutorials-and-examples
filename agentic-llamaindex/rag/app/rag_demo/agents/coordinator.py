import re
import logging
from typing import List, Dict
from .query_expansion import QueryExpander
from .validation import ValidationAgent
from .correction import CorrectionAgent

logger = logging.getLogger(__name__)

# compile this once at module load
_PAT = re.compile(
    r"Title:\s*(?P<title>.+)\n"
    r"Year:\s*(?P<year>\d+)\n"
    r"Genre:\s*(?P<genre>.+)\n"
    r"IMDb Rating:\s*(?P<rating>[\d.]+)\n"
    r"Overview:\s*(?P<overview>.+)\n"
    r"Director:\s*(?P<director>.+)\n"
    r"Stars:\s*(?P<stars>.+)",
    re.DOTALL
)

class MultiStepCoordinator:
    def __init__(
        self,
        expander: QueryExpander, 
        query_engine,
        validator: ValidationAgent,
        corrector: CorrectionAgent,
        max_steps: int = 3,
    ):
        self.expander = expander
        self.query_engine = query_engine
        self.validator = validator
        self.corrector = corrector
        self.max_steps = max_steps

    async def run(self, user_query: str) -> Dict:
        current_query = user_query
        recommendations: List[Dict] = []

        for step in range(1, self.max_steps + 1):
            logger.info(f"[Step {step}] Expanding query: {current_query}")
            expanded_list = self.expander.expand(current_query)
            logger.info(f"[Step {step}] Expanded queries: {expanded_list}")

            # Aggregate recs from all expanded queries
            aggregated_recs: List[Dict] = []
            for expanded in expanded_list:
                # call LlamaIndex query engine
                logger.info(f"[Step {step}] Querying with expanded: {expanded}")
                resp = self.query_engine.query(expanded)

                # metadata-parsing:
                recs: List[Dict] = []
                for node_with_score in resp.source_nodes:
                    text_blob = node_with_score.node.get_content()
                    m = _PAT.search(text_blob)
                    if m:
                        recs.append({
                            "title": m.group("title").strip(),
                            "imdb_rating": float(m.group("rating").strip()),
                            "overview": m.group("overview").strip(),
                            "genre": m.group("genre").strip(),
                            "released_year": int(m.group("year").strip()),
                            "director": m.group("director").strip(),
                            "stars": [s.strip() for s in m.group("stars").split(",")]
                        })
                    else:
                        logger.warning(f"[Step {step}] Failed to parse metadata from node: {text_blob[:100]}...")

                aggregated_recs.extend(recs)

            # Remove duplicates based on title
            unique_recs = list({rec["title"]: rec for rec in aggregated_recs}.values())

            logger.info(f"[Step {step}] Got {len(unique_recs)} unique recs; validating…")
            if self.validator.validate(user_query, unique_recs):
                logger.info(f"[Step {step}] Validation succeeded; stopping loop")
                recommendations = unique_recs
                break

            logger.info(f"[Step {step}] Validation failed; correcting…")
            corrected_recs = self.corrector.correct(user_query, unique_recs)
            if corrected_recs:
                logger.info(f"[Step {step}] Correction succeeded; stopping loop")
                recommendations = corrected_recs
                break

            # If correction fails, continue to next step
            current_query = f"{user_query} | corrected"

        return {
            "message": "Here are the movie recommendations after refinement:",
            "recommendations": recommendations
        }