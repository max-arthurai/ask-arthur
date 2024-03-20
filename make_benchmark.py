from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

loader = CSVLoader(file_path='arthur_index_315.csv')
data = loader.load()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

distributions = {
    simple: 0.5,
    multi_context: 0.4,
    reasoning: 0.1
}

# generate testset
testset = generator.generate_with_langchain_docs(data, 10, distributions)
testset.to_pandas().to_csv("benchmark_315.csv")
