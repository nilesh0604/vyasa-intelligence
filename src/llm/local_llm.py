from langchain_ollama import OllamaLLM


def get_llm(model: str = "llama3.2", temperature: float = 0.1):
    return OllamaLLM(
        model=model,
        base_url="http://localhost:11434",
        temperature=temperature,
    )
