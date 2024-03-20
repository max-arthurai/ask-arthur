from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.cohere import ChatCohere
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def get_langchain_llm(name: str) -> BaseLanguageModel:
    """
    https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
    """
    if 'gpt' in name:
        return ChatOpenAI(model_name=name, max_tokens=2000, temperature=0)
    elif 'claude' in name:
        return ChatAnthropic(model_name=name, max_tokens=2000, temperature=0)
    elif 'command' in name:
        return ChatCohere(model_name=name, max_tokens=2000, temperature=0)
    else:
        return HuggingFacePipeline.from_model_id(
            model_id=name,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 2000, "temperature": 0},
        )