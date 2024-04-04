import argparse
from versions.langchain_version import run as run_langchain
from versions.dspy_version import run as run_dspy
from versions.ollama_version import run as run_ollama
from versions.ollama_version import chat as chat_ollama


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval', default='chroma')
    parser.add_argument('--embedding', default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--llm', default='gpt-3.5-turbo-0125')
    parser.add_argument('--prompt', default='Give it to me short and sweet - why should I buy Arthur?')
    parser.add_argument('--framework', default='langchain')
    parser.add_argument('--mode', default='prompt')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.framework == 'langchain':
        print(run_langchain(args.prompt, args.llm, args.embedding))
    elif args.framework == 'dspy':
        run_dspy(args.prompt, args.llm, args.embedding)
    elif args.framework == 'ollama':
        if args.mode == 'chat':
            chat_ollama()
        else:
            run_ollama(args.prompt)
    else:
        raise ValueError(f"unrecognized framework: {str(args)}")
