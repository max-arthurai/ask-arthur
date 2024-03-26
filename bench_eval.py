from arthur_bench.run.testsuite import TestSuite
from arthur_bench.scoring import Scorer
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List, Optional

df = pd.read_csv("arthur_benchmark_2024-03-26.csv", index_col=0)
print(df.head())


class EmbeddingSimilarity(Scorer):

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.embedding_model = SentenceTransformer(model_name, trust_remote_code=True)

    @staticmethod
    def name() -> str:
        return "embedding_similarity"

    @staticmethod
    def requires_reference() -> bool:
        return True

    def run_batch(
        self,
        candidate_batch: List[str],
        reference_batch: Optional[List[str]] = None,
        input_text_batch: Optional[List[str]] = None,
        context_batch: Optional[List[str]] = None
    ) -> List[float]:
        outputs = [
            util.cos_sim(
                self.embedding_model.encode(x),
                self.embedding_model.encode(y)
            ).tolist()[0][0]
            for x, y in zip(candidate_batch, reference_batch)
        ]
        return outputs


suite = TestSuite(
    'scope-bench-nomic-similarity',
    EmbeddingSimilarity(),
    input_text_list=df.question.tolist(),
    reference_output_list=df.golden_answer.tolist()
)

run_names = [c for c in df.columns if c not in ["question", "golden_answer"]]
for run_name in run_names:
    suite.run(
        run_name.replace("/", "_"),
        candidate_data=df,
        candidate_column=run_name
    )
