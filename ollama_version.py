import chromadb
import json
import ollama
from transformers import AutoTokenizer

# https://docs.cohere.com/docs/prompting-command-r

tokenizer = AutoTokenizer.from_pretrained(
    "CohereForAI/c4ai-command-r-v01",
    trust_remote_code=True
)

tools = [
  {
    "name": "internet_search",
    "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
    "parameter_definitions": {
      "query": {
        "description": "Query to search the internet with",
        "type": 'str',
        "required": True
      }
    }
  },
  {
    'name': "directly_answer",
    "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
    'parameter_definitions': {}
  }
]

# print(tool_use_prompt)
# assert 0 == 1


def embed(s: str, model='nomic-embed-text'):
    return ollama.embeddings(model=model, prompt=s)['embedding']


def message(s: str, role="USER"):
    assert role.upper() in ["USER", "CHATBOT", "SYSTEM"]
    return f"<|START_OF_TURN_TOKEN|><|{role.upper()}_TOKEN|>{s}<|END_OF_TURN_TOKEN|>"


chroma_client = chromadb.PersistentClient("chroma/chroma")
collection = chroma_client.get_collection("arthur_index")
# test_prompt = 'how do I run LLM evaluations'
# k = 3
# result =
# print(result)


def run(
    prompt: str,
    llm_name: str = "command-r",
    temperature: float = 0.0,
    embedding_name: str = "nomic-embed-text",
    k: int = 5,
    verbose: bool = True
):
    conversation = [
        {"role": "system", "content": """The purpose of your job is to write search queries specifically for the AI company Arthur AI.
    Arthur AI is the AI performance company. Arthur offers four products (with integrations between the four):
    Arthur Scope is an API-first enterprise suite of tools for robust, efficient, and scalable metrics, monitoring, dashboards, and alerting.
    Arthur Shield is an API for detecting problems in LLM pipelines - hallucinations, prompt injections, leaks of personally identifiable or otherwise sensitive data, and toxic/malicious language and intent.
    Arthur Bench is an open source python package with built in UI for evaluating and comparing the components of LLM pipelines - different foundation models, prompt templates, and configurations for retrieval-augmented generation (RAG).
    Arthur Chat is a chat application that makes it easy to securely use LLMs with business data."""}
    ]

    searching = True
    while searching:
        conversation.append({"role": "user", "content": prompt})
        tool_use_prompt = tokenizer.apply_tool_use_template(
            conversation,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        tool_use_response_generator = ollama.generate(
            model=llm_name,
            options={"temperature": temperature},
            prompt=tool_use_prompt,
            stream=True,
        )

        tool_use_response = ""
        for chunk in tool_use_response_generator:
            print(chunk)
            chunk_text = chunk["response"]
            tool_use_response += chunk_text
            if verbose:
                print(chunk_text, end="", flush=True)

        try:
            # parse response
            tool_use_response = json.loads(tool_use_response.split("```json")[1].split("```")[0])
            if len(tool_use_response) == 1:
                tool_use_response = tool_use_response[0]
            else:
                print("Multiple responses!")
                print(tool_use_response)
                assert 0==1
            if tool_use_response["tool_name"] == "internet_search":
                print("starting search...")
                print("query:", tool_use_response)
                query = tool_use_response["parameters"]["query"]
                search_result = collection.query(query_embeddings=embed(query, model=embedding_name), n_results=k)
                print(search_result)
                assert 0==1
            else:
                searching = False
        except Exception as e:
            print("could not parse response")
            print(tool_use_response)
            searching = False


run("how does Arthur Bench score summaries")
