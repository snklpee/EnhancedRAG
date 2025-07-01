# app.py 
# Gradio App for Enhanced RAG 
from ast import Raise
import os
import shutil
import gradio as gr
from typing import Dict, List, Tuple

from src.ingestion.DocumentLoader import DocumentLoader
from src.ingestion.DocumentChunker import DocumentChunker
from src.ingestion.HuggingFaceEmbedder import HuggingFaceEmbedder
from src.ingestion.VectorStoreManager import VectorStoreManager

from src.generation.PromptAugmentor import PromptAugmentor
from src.generation.HuggingFaceLLM import HuggingFaceLLM
from src.generation.Prompts import Prompts
from src.generation.Fusion import FusionSummarizer

from src.utils.ModelLister import HuggingFaceModelLister
from config.settings import settings

# ——— Singletons & Globals ———
# Initializing LLMs can be time-consuming, so it's good practice to do it once.
try:
    pa_llm: HuggingFaceLLM = HuggingFaceLLM(model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
    fusion_llm: HuggingFaceLLM = HuggingFaceLLM(model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
except Exception as e:
    print(f"Error initializing LLMs: {e}")


_final_llm_cache: Dict[str, HuggingFaceLLM] = {}
loader = DocumentLoader()
augmentor = PromptAugmentor(client=pa_llm)

def get_final_llm(model_name: str) -> HuggingFaceLLM:
    """Caches and returns a HuggingFaceLLM instance for the final answer generation."""
    if model_name not in _final_llm_cache:
        print(f"Creating new LLM instance for: {model_name}")
        _final_llm_cache[model_name] = HuggingFaceLLM(model_name=model_name)
    return _final_llm_cache[model_name]

# ——— Core Application Logic ———

def handle_upload(files: list, index_name: str) -> Tuple[List[List], str]:
    """
    This function saves uploaded files to a directory
    based on the index_name, allowing the ingestion process to find them.
    """
    if not files:
        return [], "No files uploaded."
    if not index_name or not index_name.strip():
        # Enforce that an index name is provided.
        raise gr.Error("Index Name cannot be empty.")

    # Construct the target directory path and create it if it doesn't exist.
    target_dir = os.path.join(settings.CONTEXT_DIR, index_name)
    os.makedirs(target_dir, exist_ok=True)

    # Copy each uploaded file from its temporary location to the target directory.
    for file_obj in files:
        shutil.copy(file_obj.name, target_dir)

    # Prepare data for the UI DataFrame.
    df_data = [[i + 1, os.path.basename(f.name)] for i, f in enumerate(files)]
    
    return df_data, f"Uploaded {len(files)} files to index '{index_name}'."

def handle_cleanup(index_name: str) -> Tuple[str, list]:
    """
    FIX: Provides clearer feedback and cleans up the UI by clearing the file list.
    """
    if not index_name or not index_name.strip():
        return "Index Name is empty. Nothing to clear.", []
        
    target_dir = os.path.join(settings.CONTEXT_DIR, index_name)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            # UX Improvement: Also clear the file list in the UI.
            return f"Cleared all files under index '{index_name}'.", []
        except OSError as e:
            return f"Error clearing files: {e}", []
    else:
        return f"Index '{index_name}' directory does not exist.", []

def streaming_ingest(chunk_size, chunk_overlap, embed_model, index_name, ingest_state):
    """
    Performs the document ingestion process in a streaming fashion for better UI feedback.
    This function should now work correctly as handle_upload places files in the expected location.
    """
    # 0% → List files
    yield ("...", "", "", "Listing files...")
    files = loader.list_filenames(index_name)
    if not files:
        raise gr.Error(f"No files found for index '{index_name}'. Please upload files first.")
    
    # 25% → Load documents
    yield (str(len(files)), "...", "", "Loading documents...")
    docs = loader.load_documents(subdir=index_name, file_names=files)
    
    # 50% → Chunk documents
    yield (str(len(files)), str(len(docs)), "...", "Chunking documents...")
    chunker = DocumentChunker(
        hf_embedding_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.chunk_documents(docs)
    
    # 75% → Build vector index
    yield (str(len(files)), str(len(docs)), str(len(chunks)), "Embedding and indexing...")
    embeddings = HuggingFaceEmbedder(embed_model)
    vsm = VectorStoreManager(embedding_function=embeddings, index_name=index_name)
    vsm.create_index()
    vsm.add_documents(chunks)
    
    # 100% → Done
    # Store the vector store manager in the session state for the generation tab.
    ingest_state["vsm"] = vsm
    ingest_state["embedder"] = embeddings
    yield (str(len(files)), str(len(docs)), str(len(chunks)), "Ingestion Complete!")


def streaming_generate(num_prompts: int, top_k: int, llm_model: str, temperature: float,
                       max_output_tokens: int, sys_prompt_fuse: str,
                       sys_prompt_final: str, show_chunks: int, query: str, ingest_state):
    """
    Generates an answer by augmenting prompts, retrieving context, and synthesizing a final response.
    """
    vsm = ingest_state.get("vsm")
    if vsm is None:
        raise gr.Error("Ingestion not completed or failed. Please run ingestion first.")

    # 1) Generate prompts
    yield ("Generating prompts...", "", "", "")
    prompts = augmentor.generate(query=query, synthetic_count=num_prompts)
    prompts_out = "\n\n".join(f"- {p}" for p in prompts)
    
    # 2) Retrieve chunks for each prompt
    yield (prompts_out, "Retrieving documents...", "", "")
    retriever = vsm.retriever(search_type="similarity", search_kwargs={"k": top_k})
    retrieved_out = ""
    prompt_chunks = []
    for idx, p in enumerate(prompts, start=1):
        docs = retriever.invoke(p)
        prompt_chunks.append((p, docs))
        # Format retrieved chunks for display
        chunk_texts = [f"  - Doc: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Content: {doc.page_content[:100]}..." for doc in docs[:show_chunks]]
        retrieved_out += f"**Prompt {idx}:** `{p}`\n" + "\n".join(chunk_texts) + "\n\n"
        yield (prompts_out, retrieved_out, "", "")

    # 3) Generate intermediate summaries
    yield (prompts_out, retrieved_out, "Fusing summaries...", "")
    summarizer = FusionSummarizer(fusion_llm=fusion_llm, sys_prompt=sys_prompt_fuse)
    summaries = summarizer.summarize(prompt_chunks=prompt_chunks)
    summaries_out = "\n\n".join(f"**Summary {i+1}:**\n{s}" for i, s in enumerate(summaries))
    
    # 4) Generate final answer
    yield (prompts_out, retrieved_out, summaries_out, "Generating final answer...")
    final_llm = get_final_llm(llm_model)
    final_out = final_llm.get_answer(
        sys_prompt=sys_prompt_final,
        user_prompt="\n".join(summaries),
        max_tokens=max_output_tokens,
        temperature=temperature
    )
    yield (prompts_out, retrieved_out, summaries_out, final_out)


# ——— Build UI ———
try:
    lister = HuggingFaceModelLister()
    embedding_models = lister.list_models(task="sentence-similarity", filter="feature-extraction")
except Exception as e:
    print(f"Could not fetch embedding models from HuggingFace: {e}")
    embedding_models = ["sentence-transformers/all-MiniLM-L6-v2"] # Fallback

# A static list of models tested to be working (from HF inference endpoint)
hf_llms = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "meta-llama/Llama-3.1-70B-Instruct"
]

with gr.Blocks(title="Enhanced RAG App", theme=gr.themes.Soft()) as demo:
    # State object to hold session data like the vector store manager.
    session_state = gr.State({})

    gr.Markdown("## Enhanced RAG: Ingestion & Generation")

    with gr.Tab("1. Ingestion"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("#### Step 1: Upload Documents")
                index_name = gr.Textbox(label="Create a New Index Name", value="my-first-index")
                file_input = gr.File(label="Upload Documents", file_count="multiple",
                                     file_types=[".pdf", ".txt", ".md", ".py", ".json", ".html"])
                
                with gr.Row():
                    upload_btn = gr.Button("Upload Files", variant="primary")
                    cleanup_btn = gr.Button("Clear Index Files", variant="stop")

                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                files_df = gr.Dataframe(headers=["File Name"], datatype=["str"],
                                        interactive=False, label="Uploaded Files in Index")

            with gr.Column(scale=2):
                gr.Markdown("#### Step 2: Configure & Run Ingestion")
                embed_model = gr.Dropdown(label="Embedding Model", choices=embedding_models,
                                          value="sentence-transformers/all-MiniLM-L6-v2")
                with gr.Row():
                    chunk_size = gr.Number(label="Chunk Size", value=300, precision=0)
                    chunk_overlap = gr.Number(label="Chunk Overlap", value=80, precision=0)

                ingest_btn = gr.Button("Run Ingestion", variant="primary")
                ingestion_status = gr.Textbox(label="Ingestion Progress", interactive=False)
                
                with gr.Accordion(label="Ingestion Stats", open=False):
                    total_files = gr.Textbox(label="Files Found", interactive=False)
                    total_docs = gr.Textbox(label="Documents Loaded", interactive=False)
                    total_chunks = gr.Textbox(label="Chunks Created", interactive=False)

        # Link UI components to the backend functions
        upload_btn.click(
            fn=handle_upload,
            inputs=[file_input, index_name],
            outputs=[files_df, upload_status]
        )

        cleanup_btn.click(
            fn=handle_cleanup,
            inputs=[index_name],
            # UX Improvement: Clears both status and the file list.
            outputs=[upload_status, files_df]
        )

        ingest_progress_outputs = [total_files, total_docs, total_chunks, ingestion_status]
        ingest_btn.click(
            fn=streaming_ingest,
            inputs=[chunk_size, chunk_overlap, embed_model, index_name, session_state],
            outputs=ingest_progress_outputs
        )

    with gr.Tab("2. Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Step 3: Configure Generation")
                llm_model = gr.Dropdown(label="Final LLM Model", choices=hf_llms, value=hf_llms[0])
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                
                with gr.Row():
                    max_output_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                    num_prompts = gr.Number(label="# Prompts", value=3, precision=0)
                    
                with gr.Row():
                    top_k = gr.Number(label="Top-K Chunks", value=3, precision=0)
                    show_chunks = gr.Number(label="Display Chunks", value=2, precision=0)
                    
                with gr.Accordion("Advanced: System Prompts", open=False):
                    sys_fuse = gr.Textbox(label="Merge Fusion Prompt", value=Prompts.MERGE_FUSION_SYS_PROMPT, lines=5)
                    sys_final = gr.Textbox(label="Final Answer Prompt", value=Prompts.FINAL_ANS_SYS_PROMPT, lines=5)

            with gr.Column(scale=2):
                gr.Markdown("#### Step 4: Run Query")
                user_query = gr.Textbox(label="Your Question", lines=3)
                run_btn = gr.Button("Generate Answer", variant="primary")
                
                final_out = gr.Markdown(label="Final Answer")

                with gr.Accordion("Intermediate Steps", open=False):
                    prompts_out = gr.Textbox(label="Generated Prompts", lines=4, interactive=False)
                    retrieved_out = gr.Markdown(label="Retrieved Chunks")
                    summaries_out = gr.Markdown(label="Intermediate Summaries")

        run_btn.click(
            fn=streaming_generate,
            inputs=[num_prompts, top_k, llm_model, temperature, max_output_tokens, 
                    sys_fuse, sys_final, show_chunks, user_query, session_state],
            outputs=[prompts_out, retrieved_out, summaries_out, final_out]
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
