import chromadb
import json
import ollama
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")

chroma_client = chromadb.PersistentClient("data/chroma/chroma")
arthur_index = chroma_client.get_collection("arthur_index")


def generate(
    prompt: str,
    model: str = "command-r",
    start_phrase: str = "Answer: ",
    stop_phrase: str = "Grounded answer: ",
    temperature: float = 0.0,
    verbose: bool = True,
) -> str:
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
        print(chunk_text, end="", flush=True)
        if stop_phrase in response:
            response = response.split(stop_phrase)[0]
            break
    if start_phrase in response:
        response = response.split(start_phrase)[1]
    return response


def get_search_documents(s: str, k: int = 3) -> list[dict[str, str]]:
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


def command_r_tool_response(
    conversation: list[dict[str, str]],
    verbose: bool = True
) -> dict:
    """Calls command-r with the tool-use prompt template

    Args:
        conversation: list[dict[str,str]]
        verbose: bool
    Returns:
        tool_response: dict["tool_name" : str, "parameters" : dict[str , str]]
    """
    tools = [{
        "name": "internet_search",
        "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
        "parameter_definitions": {
            "query": {
                "description": "Query to search the internet with",
                "type": 'str',
                "required": True
            }
        }
    }, {
        "name": "directly_answer",
        "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation \
        history",
        "parameter_definitions": {}
    }]
    tools_prompt = tokenizer.apply_tool_use_template(
        conversation,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    if verbose:
        print("choosing tool")
    tools_response = generate(tools_prompt, verbose=verbose)
    tools_response = json.loads(tools_response.split("```json")[1].split("```")[0])
    return tools_response


def command_r_chat_response(
    conversation: list[dict[str, str]],
    verbose=True
) -> str:
    """Calls command-r with the chat prompt template

    Args:
        conversation: list[dict[str:str]]
        verbose: bool
    Returns:
        response: str
    """
    response_prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    response = generate(response_prompt, verbose=verbose)
    return response


def command_r_grounded_response(
    conversation: list[dict[str, str]],
    documents: list[dict[str, str]],
    verbose=True
) -> str:
    """Calls command-r with the tool-use prompt template

    Args:
        conversation: list[dict[str:str]]
        documents: list[dict[str:str]]
        verbose: bool
    Returns:
        tool_response: dict["tool_name" : str, "parameters" : dict[str : str]]
    """
    response_prompt = tokenizer.apply_grounded_generation_template(
        conversation,
        documents=documents,
        citation_mode="accurate",
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(response_prompt, verbose=verbose)


def run_conversation(conversation: list[dict[str,str]], verbose=False) -> str:
    """Runs the command-r model via Ollama on a conversation

    Args:
        conversation: list[dict[str,str]]
        verbose: bool
    Returns:
        response: str
    """
    tools_response = command_r_tool_response(conversation, verbose=verbose)
    if not any(x["tool_name"] == "internet_search" for x in tools_response):
        return command_r_chat_response(conversation, verbose=verbose)
    else:
        documents = []
        for x in tools_response:
            if x["tool_name"] == "internet_search":
                documents.extend(get_search_documents(x["parameters"]["query"]))
        return command_r_grounded_response(conversation, documents, verbose=verbose)


def run(prompt: str, verbose=False) -> str:
    """Runs the command-r model via Ollama on a single prompt

    Args:
        prompt: str
        verbose: bool
    Returns:
        response: str
    """
    return run_conversation([{"role": "user", "content": prompt}], verbose=verbose)


def chat():
    """Runs a chat session with command-r via Ollama"""
    running = True
    conversation = [{"role": "system", "content": "You are a helpful AI assistant."}]
    while running:
        user_input = input(">>>")
        conversation.append({"role": "user", "content": user_input})
        response = run_conversation(conversation, verbose=True)
        conversation.append({"role": "assistant", "content": response})
