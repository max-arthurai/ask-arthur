import argparse
from langchain_version import run as run_langchain
from dspy_version import run as run_dspy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval', default='chroma-nomic')
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--llm', default='gpt-3.5-turbo-0125')
    parser.add_argument('--prompt', default='Give it to me short and sweet - why should I buy Arthur?')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    # run_langchain(args)
    run_dspy(args)
