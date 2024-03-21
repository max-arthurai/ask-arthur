from dsp.utils import deduplicate
from dsp.modules.anthropic import Claude
import dspy
from dspy.retrieve import chromadb_rm
from sentence_transformers import SentenceTransformer


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="question about Arthur AI")
    queries_so_far = dspy.InputField(desc="the queries asked so afr")
    query = dspy.OutputField(desc="""Search query for the Arthur website & documentation 
Search query should help answer the question or gather related info, and be different from previous queries""")


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
        """
        For each hop, generate a new search query and extend the context with retrieved passages
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
                print("!!! query:", query, "!!!")
                print("context", context)
        answer = self.generate_answer(context=context, question=question, config=dict(max_tokens=2000)).answer
        return dspy.Prediction(context=context, answer=answer)


def configure_dspy_settings(args):
    """
    Sets the LLM and retrieval config for all DSPy calls
    """
    if "gpt" in args.llm:
        lm = dspy.OpenAI(model=args.llm)
    elif "claude" in args.llm:
        lm = Claude(model=args.llm)
    else:
        raise ValueError("use openai or anthropic dawg trust me")

    if args.retrieval == "chroma-nomic":
        nmc = SentenceTransformer(
            model_name_or_path="nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        nmc_embed = lambda l : [nmc.encode(x).tolist() for x in l]
        rm = chromadb_rm.ChromadbRM(
            collection_name="arthur_index",
            persist_directory="chroma/chroma",
            embedding_function=nmc_embed
        )
    else:
        raise ValueError("use nomic dawg trust me")
    dspy.settings.configure(lm=lm, rm=rm)


def run(args):
    configure_dspy_settings(args)
    mh = MultiHop()
    prediction = mh(question=args.prompt)
    print("\n\n\nQuestion:", args.prompt, "\n\n Ask Arthur Answer:\n\n", prediction.answer)
