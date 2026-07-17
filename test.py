
import json

# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama

llm=ChatOllama(model="qwen2.5:3b")
response=llm.invoke("Hello! How are you?")   
print("=" * 60)
print("LLM Response")
print("=" * 60)
print(response.content)

print("\nMetadata")
print("=" * 60)
print(json.dumps(response.model_dump(), indent=4))