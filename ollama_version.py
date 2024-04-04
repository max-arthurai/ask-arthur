import chromadb
import json
import ollama
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")

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

chroma_client = chromadb.PersistentClient("chroma/chroma")
arthur_index = chroma_client.get_collection("arthur_index")


def generate(
    prompt: str,
    model: str = "command-r",
    start_phrase: str = "Answer: ",
    stop_phrase: str = "Grounded answer: ",
    temperature: float = 0.0,
    verbose: bool = True,
):
    """Responds with the command-r model via the Ollama local API

    Args:
        prompt: str
        model: str
        start_phrase: str
        stop_phrase: str
        temperature: float
        verbose: bool
    Returns:
        documents: list[{"title": ..., "text": ...}]
    """
    if verbose:
        print(" ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^")
        print("   Prompt:\n", prompt)
    generator = ollama.generate(
        model=model,
        options={"temperature": temperature},
        prompt=prompt,
        stream=True,
    )
    response = ""
    for chunk in generator:
        chunk_text = chunk["response"]
        response += chunk_text
        if verbose:
            print(chunk_text, end="", flush=True)
        if stop_phrase in response:
            response = response.split(stop_phrase)[0]
            break
    if start_phrase in response:
        response = response.split(start_phrase)[1]
    return response


def get_search_documents(s: str, k: int = 3):
    """Uses the Nomic embedding model via the Ollama local API to search for relevant documents from arthur_index

    Args:
        s: str, search query
        k: int, number of search results to return
    Returns:
        documents: list[{"title": ..., "text": ...}]
    """
    search_result = arthur_index.query(
        query_embeddings=ollama.embeddings(model="nomic-embed-text", prompt=s)['embedding'],
        where={"content_type": {"$ne": "arthur_blog"}},
        n_results=k
    )
    documents = [
        {"title": title, "text": text}
        for title, text in zip(
            [x['source'] for x in search_result['metadatas'][0]],
            search_result['documents'][0]
        )
    ]
    return documents


def command_r_respond(conversation: list[dict[str:str]], verbose=True):
    """Responds to a conversation"""
    tool_prompt = tokenizer.apply_tool_use_template(
        conversation,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    if verbose: print("choosing tool")
    tool_response = generate(tool_prompt, verbose=False)
    tool_response = json.loads(tool_response.split("```json")[1].split("```")[0])
    tool_name = tool_response[0]["tool_name"]
    if tool_name == "directly_answer":
        response_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif tool_name == "internet_search":
        query = tool_response[0]["parameters"]["query"]
        documents = get_search_documents(query)
        response_prompt = tokenizer.apply_grounded_generation_template(
            conversation,
            documents=documents,
            citation_mode="accurate",
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        raise ValueError("bad tool name :/")
    response = generate(response_prompt, verbose=True)
    return response


def run(prompt: str):
    system_prompt = """The purpose of your job is to write search queries specifically for the AI company Arthur AI.
Arthur AI is the AI performance company. Arthur offers four products (with integrations between the four):
Arthur Scope is an API-first enterprise suite of tools for robust, efficient, and scalable metrics, monitoring, dashboards, and alerting.
Arthur Shield is an API for detecting problems in LLM pipelines - hallucinations, prompt injections, leaks of personally identifiable or otherwise sensitive data, and toxic/malicious language and intent.
Arthur Bench is an open source python package with built in UI for evaluating and comparing the components of LLM pipelines - different foundation models, prompt templates, and configurations for retrieval-augmented generation (RAG).
Arthur Chat is a chat application that makes it easy to securely use LLMs with business data."""
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    response = command_r_respond(conversation)
    return response
