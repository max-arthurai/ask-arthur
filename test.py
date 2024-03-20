import argparse
from langchain.chains.chains import get_chain
from langchain.generators.generators import get_langchain_llm
from langchain.retrievers.retrievers import get_langchain_ensemble_retriever

def test_langchain():
    pass
def test_dspy():
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', default='nomic-ai/nomic-embed-text-v1.5')
    parser.add_argument('--k', default=30, type=int)
    parser.add_argument('--llm', default='gpt-3.5-turbo-0125')
    parser.add_argument('--prompt', default='Any Arthur classification metrics besides accuracy?')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    retriever = get_langchain_ensemble_retriever(
        f"retrievers/arthur_faiss_{args.embedding}",
        bm25k=args.k//2,
        faissk=args.k//2
    )

    llm = get_langchain_llm(args.llm)

    chain = get_chain(retriever, llm)
    for text in chain.stream(args.prompt):
        print(text, end='', flush=True)
