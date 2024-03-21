import argparse
from langchain_version import run as run_langchain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', default='nomic-ai/nomic-embed-text-v1.5')
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--llm', default='gpt-4-0125-preview')
    parser.add_argument('--prompt', default='Give it to me short and sweet - why should I buy Arthur?')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    run_langchain(args)
