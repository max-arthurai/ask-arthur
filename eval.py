from datetime import date
import pandas as pd
from langchain_version import run

try:
    df = pd.read_csv(f"arthur_benchmark_{date.today()}.csv", index_col=0)
except NameError:
    df = pd.read_csv(f"arthur_benchmark.csv", index_col=0)


def run_benchmark(llm: str, embedding: str, k: int):
    run_name = f"{llm}+{embedding}@{k}"
    print("\n\n\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", run_name)
    if run_name not in df.columns:
        for i, row in df.iterrows():
            print("~~~~~\n>>>", row.question)
            answer = run(row.question, llm_name=llm, embedding_name=embedding, k=k)
            print(answer)
            df.loc[i, run_name] = answer
        df.to_csv(f"arthur_benchmark_{date.today()}.csv")


llm_names = [
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229"
]
embedding_names = [
    "sentence-transformers/all-MiniLM-L12-v2",
    "nomic-ai/nomic-embed-text-v1.5"
]
retrieval_ks = [1, 3, 5, 7]
for llm_name in llm_names:
    for embedding_name in embedding_names:
        for retrieval_k in retrieval_ks:
            run_benchmark(llm_name, embedding_name, retrieval_k)
