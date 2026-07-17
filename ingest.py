# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pyrefly: ignore [missing-import]
from langchain_ollama import OllamaEmbeddings


pdf_path = r"data\policy.pdf"  
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Total Pages:{len(documents)}")

# display the first page
# Display each page
for doc in documents:
    print("=" * 50)
    print("Page:", doc.metadata["page"] + 1)
    print(doc.page_content)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,  chunk_overlap=100)

# split the loaded pages    
chunks = text_splitter.split_documents(documents)

# count the chunks
print(f"Total Chunks:{len(chunks)}")
print("\nFirst Chunk")
print("="*50)
print(chunks[0].page_content)
print("\nChunk metadata")
print("="*50)
print(chunks[0].metadata)

# display chunk by chunk
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("="*50)
    print(chunk.page_content)

# Create the Embedding Model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("Hello, how are you?") 

print("\nVector")
print("="*50)
print(len(vector))

# Generate Embeddings for All Chunks
chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])

print("\nChunk Vectors")
print("="*50)
print(len(chunk_vectors))

# Check how many vectors were created
print("Total Chunks:", len(chunks))
print("\nTotal Embeddings:", len(chunk_vectors))

# Print the beginning of the first embedding
print(chunk_vectors[0][:5]) 