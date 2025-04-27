# app.py

import os
import shutil
import shutil
import glob
import gradio as gr
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.HuggingFaceEmbedder import HuggingFaceEmbedder
from src.ingestion.VectorStoreManager import VectorStoreManager

from src.generation.PromptAugmentor import PromptAugmentor
from src.generation.HuggingFaceLLM import HuggingFaceLLM
from src.generation.Prompts import Prompts

from src.utils.ModelLister import HuggingFaceModelLister

from config.settings import settings


# ——— Global/shared singletons & caches ———
pa_llm = HuggingFaceLLM(model_name="meta-llama/Llama-3.1-8B-Instruct")
fusion_llm = HuggingFaceLLM(model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
_final_llm_cache: Dict[str, HuggingFaceLLM] = {}

def get_final_llm(model_name: str) -> HuggingFaceLLM:
    if model_name not in _final_llm_cache:
        _final_llm_cache[model_name] = HuggingFaceLLM(model_name=model_name)
    return _final_llm_cache[model_name]

# ——— Helpers ———
def upload_files(uploaded_files, index_name):
    """
    Saves uploaded files to settings.CONTEXT_DIR/index_name, and returns:
      1) A list of [sr_no, filename] rows for the DataFrame.
      2) A status message for the Textbox.
    """
    try:
        save_dir = os.path.join(settings.CONTEXT_DIR, index_name)
        os.makedirs(save_dir, exist_ok=True)
        rows = []
        for i, f in enumerate(uploaded_files, start=1):
            filename = os.path.basename(f.name)
            dest = os.path.join(save_dir, filename)
            shutil.copyfile(f.name, dest)
            rows.append([i, filename])
        return rows, "Upload successful"
    except Exception as e:
        # Return empty rows and the error message
        return [], f"Upload failed: {e}"


def cleanup_index(index_name: str):
    """Deletes the persisted index directory."""
    dirpath = os.path.join(settings.CONTEXT_DIR, index_name)
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    return f"Cleared index `{index_name}`."


def format_prompts_and_chunks(
    prompt_chunks: List[Tuple[str, List[Document]]],
    max_chunks: int = None
) -> Tuple[str, List[str]]:
    retrieved_parts = []
    contexts = []
    for p_idx, (prompt, chunks) in enumerate(prompt_chunks, start=1):
        # Retrieved snippet
        lines = [f"**Prompt {p_idx}:** {prompt}\n"]
        for c_idx, doc in enumerate(chunks[:max_chunks] if max_chunks else chunks, start=1):
            lines.append(f"- Chunk {c_idx}: {doc.page_content}")
        retrieved_parts.append("\n".join(lines))

        # Full context
        ctx = [f"Query: {prompt}"]
        for c_idx, doc in enumerate(chunks, start=1):
            ctx.append(f"Chunk {c_idx}: {doc.page_content}")
        contexts.append("\n".join(ctx))

    return "\n\n".join(retrieved_parts), contexts


def ingest_pipeline(
    chunk_size, chunk_overlap, embed_model, index_name, ingest_state, progress=gr.Progress()
):
    progress(0, desc="Listing uploaded files…")
    files = loader.list_filenames(index_name)
    if not files:
        raise gr.Error(f"No files found under index `{index_name}`. Please upload first.")
    progress(0.2, desc="Loading documents…")
    documents = loader.load_documents(subdir=index_name, file_names=files)
    progress(0.4, desc="Chunking documents…")
    chunker = DocumentChunker(
        hf_embedding_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.chunk_documents(documents)
    progress(0.6, desc="Creating vector index…")
    embeddings = HuggingFaceEmbedder(embed_model)
    vsm = VectorStoreManager(embedding_function=embeddings, index_name=index_name)
    vsm.create_index()
    progress(0.8, desc="Adding embeddings…")
    vsm.add_documents(chunks)
    progress(1, desc="Done.")

    # Persist for this session
    ingest_state["vsm"] = vsm
    ingest_state["embedder"] = embeddings

    return str(len(files)), str(len(documents)), str(len(chunks))


def generate_pipeline(
    num_prompts, top_k, llm_model, temperature,
    max_output_tokens, sys_prompt_fuse,
    sys_prompt_final, show_chunks, query, ingest_state, progress=gr.Progress()
):
    # Input validation
    if num_prompts < 1:
        raise gr.Error("Number of prompts must be ≥ 1.")
    if top_k < 1:
        raise gr.Error("Top K chunks must be ≥ 1.")

    vsm = ingest_state.get("vsm")
    if vsm is None:
        raise gr.Error("No index found—please run ingestion first.")

    progress(0, desc="Generating synthetic prompts…")
    augmentor = PromptAugmentor(client=pa_llm)
    prompts = augmentor.generate(query=query, synthetic_count=num_prompts)

    progress(0.2, desc="Retrieving top-K chunks…")
    retriever = vsm.retriever(search_type="similarity", search_kwargs={"k": top_k})
    prompt_chunks = [(p, retriever.invoke(p)) for p in prompts]

    progress(0.4, desc="Formatting retrieved chunks…")
    retrieved, contexts = format_prompts_and_chunks(prompt_chunks, max_chunks=show_chunks)

    progress(0.6, desc="Generating intermediate summaries…")
    summaries = []
    for idx, ctx in enumerate(contexts, start=1):
        summ = fusion_llm.get_answer(
            sys_prompt=sys_prompt_fuse,
            user_prompt=ctx,
            temperature=0.3,
            max_tokens=300
        )
        summaries.append(f"**Prompt {idx} Summary:** {summ}")
    summaries_text = "\n\n".join(summaries)

    progress(0.8, desc="Generating final answer…")
    final_llm = get_final_llm(llm_model)
    final_answer = final_llm.get_answer(
        sys_prompt=sys_prompt_final,
        user_prompt=summaries_text,
        max_tokens=max_output_tokens,
        temperature=temperature
    )
    progress(1, desc="Done.")
    prompts_out    = "\n\n".join(prompts) 
    
    return prompts_out, retrieved, summaries_text, final_answer


# ——— Build UI ———
lister = HuggingFaceModelLister()
embedding_models = lister.list_models(task="sentence-similarity", filter="feature-extraction")
hf_llms = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
]

loader = DocumentLoader()


def build_ingestion_tab():
    with gr.Tab("Ingestion"):
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".py", ".json", ".rs", ".cpp"]
                )
                files_df = gr.Dataframe(
                    headers=["Sr. No", "File Name"],
                    datatype=["number", "str"],
                    interactive=False,
                    label="Files Uploaded"
                )
                upload_btn = gr.Button("Upload Files")
                cleanup_btn = gr.Button("Clear Uploaded Files")
                upload_status = gr.Textbox(label="Status", interactive=False)
                cleanup_status = gr.Textbox(label="Cleanup Status", interactive=False)

            with gr.Column(scale=1):
                embed_model = gr.Dropdown(
                    filterable=True,
                    label="Embedding Model",
                    choices=embedding_models,
                    value="sentence-transformers/all-mpnet-base-v2"
                )
                index_name = gr.Textbox(label="Vector Index Name", value="index")
                chunk_size = gr.Number(label="Chunk Size", value=300, precision=0)
                chunk_overlap = gr.Number(label="Chunk Overlap", value=80, precision=0)
                ingest_btn = gr.Button("Run Ingestion")

        upload_btn.click(fn=upload_files,
                         inputs=[file_input, index_name],
                         outputs=[files_df, upload_status])
        cleanup_btn.click(fn=cleanup_index,
                          inputs=index_name,
                          outputs=cleanup_status)

        with gr.Accordion("Ingestion Outputs", open=False):
            total_files = gr.Textbox(label="Total Files")
            total_docs = gr.Textbox(label="Total Documents")
            total_chunks = gr.Textbox(label="Total Chunks")

        ingest_btn.click(fn=ingest_pipeline,
                         inputs=[chunk_size, chunk_overlap, embed_model, index_name, session_state],
                         outputs=[total_files, total_docs, total_chunks])

    return locals()


def build_generation_tab():
    with gr.Tab("Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                llm_model = gr.Dropdown(label="LLM Model", choices=hf_llms, value=hf_llms[-1])
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                max_output_tokens = gr.Number(label="Max Output Tokens", value=256, precision=0)
                num_prompts = gr.Number(label="Number of Prompts", value=1, precision=0)
                top_k = gr.Number(label="Top-K Chunks", value=3, precision=0)
                show_chunks = gr.Number(label="Chunks to Display", value=2, precision=0)

                with gr.Accordion("System Prompts", open=False):
                    sys_fuse = gr.Textbox(label="Merge Fusion Prompt",
                                          value=Prompts.MERGE_FUSION_SYS_PROMPT,lines=4)
                    sys_final = gr.Textbox(label="Final Answer Prompt",
                                           value=Prompts.FINAL_ANS_SYS_PROMPT,lines=4)

            with gr.Column(scale=1):
                user_query = gr.Textbox(label="User Query")
                run_btn = gr.Button("Run Generation")

                with gr.Accordion("Intermediate Outputs", open=False):
                    with gr.Row():
                        prompts_out = gr.Textbox(label="Augmented Prompts", value="Your Generated Prompts will Appear here", lines=4)
                    with gr.Row():
                        retrieved_out = gr.Markdown(label="Retrieved Chunks", value="A subset of your `top-k` chunks will be shown here")
                    with gr.Row():
                        summaries_out = gr.Markdown(label="Summaries", value="Intermediate summaries will appear here…") 

                final_out = gr.Textbox(label="Final Answer", value="The final answer will appear here…", lines=4)

        run_btn.click(fn=generate_pipeline,
                      inputs=[num_prompts, top_k, llm_model, temperature,
                              max_output_tokens, sys_fuse, sys_final,
                              show_chunks, user_query, session_state],
                      outputs=[prompts_out, retrieved_out, summaries_out, final_out])

    return locals()


with gr.Blocks(title="Enhanced RAG App") as demo:
    session_state = gr.State({})
    gr.Markdown("## Enhanced RAG: Ingestion & Generation")
    build_ingestion_tab()
    build_generation_tab()

if __name__ == "__main__":
    demo.launch()
