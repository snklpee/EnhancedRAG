# Enhanced RAG (Retrieval-Augmented Generation)

A sophisticated RAG system that implements prompt augmentation and fusion techniques to improve the quality and accuracy of document-based question answering. The system uses HuggingFace models for embeddings and language generation, with a user-friendly Gradio interface.

## 🚀 Features

- **Prompt Augmentation**: Generates multiple synthetic prompts from a single query to improve retrieval diversity
- **Fusion Summarization**: Creates intermediate summaries from retrieved document chunks before final answer generation
- **Multi-Document Support**: Handles `.pdf`, `.txt`, `.md`, `.py`, `.json`, and `.html` files
- **Semantic Chunking**: Intelligent document chunking with configurable overlap and size
- **Vector Store Management**: FAISS-based vector storage for efficient similarity search
- **Interactive Web UI**: Gradio-based interface for easy document upload and querying
- **Configurable LLMs**: Support for multiple HuggingFace language models
- **Real-time Progress**: Streaming updates during ingestion and generation processes

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🏗️ Architecture

The Enhanced RAG system consists of three main stages:

1. **Document Ingestion**
   - Load documents from various formats
   - Chunk documents using semantic-aware splitting
   - Generate embeddings using HuggingFace models
   - Store in FAISS vector database

2. **Prompt Augmentation**
   - Generate multiple synthetic prompts from user query
   - Retrieve relevant document chunks for each prompt
   - Create diverse retrieval contexts

3. **Fusion Generation**
   - Summarize retrieved chunks for each prompt
   - Combine summaries into final coherent answer
   - Use configurable system prompts for control

## 📦 Installation

### Prerequisites

- Python 3.11+
- HuggingFace account and API token

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/snklpee/EnhancedRAG.git
   cd EnhancedRAG
   ```

2. **Install dependencies**
For pip:
   ```bash
   pip install -r requirements.txt
   ```
For uv:
   ```bash
   uv pip install -r requirements.txt
   ```
3. **Set up environment variables**
   ```bash
   cp config/.env.example config/.env
   ```
   
   Edit `config/.env` and add your credentials:
   ```env
   HF_TOKEN=<your_huggingface_token_here>
   CONTEXT_DIR=context
   FAISS_INDEXES=context/faiss_indexes
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token (required) | - |
| `CONTEXT_DIR` | Directory for uploaded documents | `context` |
| `FAISS_INDEXES` | Directory for FAISS vector stores | `context/faiss_indexes` |

### Model Configuration

The system supports various HuggingFace models:

**Embedding Models** (automatically fetched):
- `sentence-transformers/all-MiniLM-L6-v2` (default)
- Other sentence-transformer models

**Language Models**:
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`
- `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`
- `meta-llama/Llama-3.1-70B-Instruct`

`These models were tested to be working with Huggingface Inference at the time of creation of this doc, if you find that these are not working for you; check if there are any Inference providers for the model you've selected and ensure that your Huggingface Account and the Token used in this project have the necessary permissions.`

## 🚀 Usage

### Starting the Application

```bash
gradio app.py
```

The Gradio interface will be available at `http://localhost:7860`

### Using the Web Interface

#### 1. Document Ingestion

1. **Upload Documents**:
   - Create a unique index name
   - Upload one or more documents (PDF, TXT, MD, PY, JSON, HTML)
   - Click "Upload Files"

2. **Configure Ingestion**:
   - Select embedding model
   - Set chunk size (default: 300)
   - Set chunk overlap (default: 80)
   - Click "Run Ingestion"

#### 2. Question Answering

1. **Configure Generation**:
   - Select final LLM model
   - Adjust temperature (0.0-1.0)
   - Set max output tokens
   - Configure number of synthetic prompts
   - Set top-K chunks to retrieve

2. **Ask Questions**:
   - Enter your question in the query box
   - Click "Generate Answer"
   - View intermediate steps (prompts, chunks, summaries)

### Programmatic Usage

```python
from src.ingestion.DocumentLoader import DocumentLoader
from src.ingestion.DocumentChunker import DocumentChunker
from src.ingestion.HuggingFaceEmbedder import HuggingFaceEmbedder
from src.ingestion.VectorStoreManager import VectorStoreManager
from src.generation.PromptAugmentor import PromptAugmentor
from src.generation.HuggingFaceLLM import HuggingFaceLLM

# Initialize components
loader = DocumentLoader()
llm = HuggingFaceLLM(model_name="meta-llama/Llama-3.1-8B-Instruct")
augmentor = PromptAugmentor(client=llm)

# Load and process documents
docs = loader.load_documents(subdir="my-index")
chunker = DocumentChunker(hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2")
chunks = chunker.chunk_documents(docs)

# Create vector store
embeddings = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
vsm = VectorStoreManager(embedding_function=embeddings, index_name="my-index")
vsm.create_index()
vsm.add_documents(chunks)

# Generate augmented prompts and retrieve
prompts = augmentor.generate("What is the main topic?", synthetic_count=3)
retriever = vsm.retriever(search_type="similarity", search_kwargs={"k": 5})
```

## 📁 Project Structure

```
EnhancedRAG/
├── app.py                      # Main Gradio application
├── requirements.txt            # Python dependencies
├── config/
│   ├── settings.py            # Configuration management
│   └── .env.example           # Environment variables template
├── src/
│   ├── ingestion/             # Document processing modules
│   │   ├── DocumentLoader.py   # File loading utilities
│   │   ├── DocumentChunker.py  # Text chunking logic
│   │   ├── HuggingFaceEmbedder.py # Embedding generation
│   │   └── VectorStoreManager.py # FAISS vector store operations
│   ├── generation/            # Answer generation modules
│   │   ├── PromptAugmentor.py  # Query augmentation
│   │   ├── HuggingFaceLLM.py   # LLM interface
│   │   ├── Fusion.py           # Summary fusion
│   │   └── Prompts.py          # System prompt templates
│   ├── utils/                 # Utility modules
│   │   ├── ModelLister.py      # HuggingFace model discovery
│   │   └── metrics.py          # Evaluation metrics
│   └── evaluation/            # Performance evaluation
│       └── CorePerfEval.py     # Core performance metrics
├── docs/                      # Documentation
└── context/                   # Document storage (created at runtime)
```

## 🔧 API Reference

### Key Classes

#### `DocumentLoader`
Handles loading documents from various file formats.

```python
loader = DocumentLoader()
docs = loader.load_documents(subdir="index_name", file_names=["file1.pdf"])
```

#### `PromptAugmentor`
Generates synthetic prompts to improve retrieval diversity.

```python
augmentor = PromptAugmentor(client=llm_instance)
prompts = augmentor.generate(query="question", synthetic_count=3)
```

#### `FusionSummarizer`
Creates intermediate summaries from retrieved document chunks.

```python
summarizer = FusionSummarizer(fusion_llm=llm, sys_prompt=system_prompt)
summaries = summarizer.summarize(prompt_chunks=[(prompt, docs)])
```

#### `VectorStoreManager`
Manages FAISS vector store operations.

```python
vsm = VectorStoreManager(embedding_function=embeddings, index_name="my-index")
vsm.create_index()
vsm.add_documents(document_chunks)
retriever = vsm.retriever(search_type="similarity", search_kwargs={"k": 5})
```

### Configuration Parameters

#### Chunking Parameters
- `chunk_size`: Size of text chunks (default: 300)
- `chunk_overlap`: Overlap between chunks (default: 80)

#### Generation Parameters
- `temperature`: Sampling temperature (0.0-1.0)
- `max_tokens`: Maximum output tokens
- `top_k`: Number of chunks to retrieve
- `synthetic_count`: Number of synthetic prompts to generate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black src/ app.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for providing the language models and embeddings
- LangChain for document processing utilities
- Gradio for the web interface framework
- FAISS for efficient similarity search

## 📞 Support

For questions or issues, please:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Note**: This project requires a HuggingFace API token for accessing language models. Make sure to set up your credentials before running the application.
