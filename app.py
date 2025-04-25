from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.HuggingFaceEmbedder import HuggingFaceEmbedder

loader = DocumentLoader()

docs = loader.load_directory("pdfs")
# print(type(docs[0].page_content))

chunker = DocumentChunker(
    hf_tokenizer_name="sentence-transformers/all-mpnet-base-v2",
    chunk_size=300,
    chunk_overlap=80
)

chunks, token_count = chunker.chunk_documents(docs)
print(chunks[0].metadata)
print("Total chunks produced: ", len(chunks))
print(len(chunks)," chunks worth ",token_count," tokens")
print("data type of chunks",type(chunks[0]))
print("--"*30)

embedder = HuggingFaceEmbedder("sentence-transformers/all-mpnet-base-v2")
print("initialized embedder")
v1, dim = embedder.embed_query(chunks[0].page_content)
print("dimension",dim)

