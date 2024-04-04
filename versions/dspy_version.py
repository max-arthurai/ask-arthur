from dsp.utils import deduplicate
from dsp.modules.anthropic import Claude
import dspy
from dspy.retrieve import chromadb_rm
from sentence_transformers import SentenceTransformer


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="question about Arthur AI")
    queries_so_far = dspy.InputField(desc="the queries asked so far")
    query = dspy.OutputField(desc="""Search query for the Arthur website & documentation 
Search query should help answer the question or gather related info, and be different from the queries asked so far""")


class GenerateAnswer(dspy.Signature):
    """Dense technical answer about Arthur AI for a public customer-facing chatbot"""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="question about Arthur AI")
    answer = dspy.OutputField(desc="""Answer for the question about Arthur AI. Make this extremely dense and technical.
May include code only if the code is in the context.""")


class MultiHop(dspy.Module):
    """Runs RAG with multiple rounds of context retrieval"""

    def __init__(self, passages_per_hop=10, max_hops=3):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question, verbose=True):
        """
        Generate a query and retrieve new context for each hop in self.max_hops
        Then answer the question
        """
        context = []
        queries_so_far = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](
                context=context,
                question=question,
                queries_so_far=str(queries_so_far)
            ).query
            queries_so_far.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
            if verbose:
                print("<query>", query, "</query>")
        answer = self.generate_answer(context=context, question=question, config=dict(max_tokens=2000)).answer
        return dspy.Prediction(context=context, answer=answer)


def configure_dspy_settings(llm_name, embedding_name, retrieval="chroma"):
    """
    Sets the LLM and retrieval config for all DSPy calls
    """
    if "gpt" in llm_name:
        lm = dspy.OpenAI(model=llm_name)
    elif "claude" in llm_name:
        lm = Claude(model=llm_name)
    else:
        raise ValueError("use openai or anthropic dawg trust me")

    assert retrieval == "chroma"  # todo allow other options
    embedding_model = SentenceTransformer(
        model_name_or_path=embedding_name,
        trust_remote_code=True
    )

    def embed(texts: list[str]) -> list[list[float]]:
        return [embedding_model.encode(x).tolist() for x in texts]
    rm = chromadb_rm.ChromadbRM(
        collection_name="arthur_index",
        persist_directory="chroma/chroma",
        embedding_function=embed
    )
    dspy.settings.configure(lm=lm, rm=rm)


def run(prompt, llm_name="gpt-4-0125.preview", embedding_name="nomic-ai/nomic-embed-text-v1.5"):
    configure_dspy_settings(llm_name, embedding_name)
    mh = MultiHop()
    prediction = mh(question=prompt)
    print("\n\n\nQuestion:", prompt, "\n\n Ask Arthur Answer:\n\n", prediction.answer)
