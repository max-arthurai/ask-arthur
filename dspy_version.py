from dsp.utils import deduplicate
from dsp.modules.anthropic import Claude
import dspy
from dspy.retrieve import chromadb_rm
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Union


nmc = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={'trust_remote_code': True}
)


def get_nomic_embedding(s: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    if isinstance(s, list):
        return [get_nomic_embedding(s_) for s_ in s]
    return nmc.embed_query(s)


dspy.settings.configure(
    # lm=dspy.OpenAI(
    #     model="gpt-4-0125-preview"
    # ),
    lm=Claude(
        model="claude-3-opus-20240229"
    ),
    rm=chromadb_rm.ChromadbRM(
        collection_name="arthur_index",
        persist_directory="../chroma/chroma",
        embedding_function=get_nomic_embedding
    )
)


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="question about Arthur AI")
    query = dspy.OutputField(desc="""Search query for the Arthur website & documentation 
Search query should help answer the question or gather related info""")


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="question about Arthur AI")
    answer = dspy.OutputField(desc="""Answer for the question about Arthur AI, often between 1 and 5 sentences.
May include code only if the code is in the context.""")


class MultiHop(dspy.Module):
    """Runs RAG with two rounds of context retrieval"""

    def __init__(self, passages_per_hop=10, max_hops=3):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question, verbose=True):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
            if verbose:
                print("!!! query:", query, "!!!")
                print("context", context)
        answer = self.generate_answer(context=context, question=question, config=dict(max_tokens=2000)).answer
        return dspy.Prediction(context=context, answer=answer)


if __name__ == "__main__":
    mh = MultiHop()
    question = """I am a data scientist who is using a mix of random forest classifiers and LLMs
Which Arthur products should I use for my different model types?"""
    prediction = mh(question=question)
    print("\n\n\nQuestion:", question, "\n\n Ask Arthur Answer:\n\n", prediction.answer)
