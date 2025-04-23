from src.ingestion.loader import DocumentLoader

loader = DocumentLoader()

docs = loader.load_documents("pdfs")
print(type(docs[0]))
