# app.py

import os
import shutil
import gradio as gr

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

# Model lists
lister = HuggingFaceModelLister()
embedding_models = lister.list_models(task="sentence-similarity", filter="feature-extraction")
hf_llms=[
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"]

loader = DocumentLoader()

pa_llm = HuggingFaceLLM(model_name="meta-llama/Llama-3.1-8B-Instruct")    
fusion_llm = HuggingFaceLLM(model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")


def upload_files(uploaded_files, index_name):
    """
    Saves uploaded files to settings.CONTEXT_DIR/index_name, and returns
    a list of [sr_no, filename] rows for the DataFrame.
    """
    save_dir = os.path.join(settings.CONTEXT_DIR, index_name)
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for i, f in enumerate(uploaded_files, start=1):
        # f.name is the temporary file path on disk
        filename = os.path.basename(f.name)
        dest = os.path.join(save_dir, filename)
        shutil.copyfile(f.name, dest)
        rows.append([i, filename])
    return rows

def format_prompts_and_chunks(
    prompt_chunks: list[tuple[str, list[Document]]],
    max_chunks: int | None = None
) -> tuple[str, list[str]]:
    """
    - Builds a '\n\n'-joined 'retrieved' block.
    - Returns a list of 'contexts' (just prompt+all chunks) for summaries.
    """
    retrieved_parts = []
    contexts = []

    for p_idx, (prompt, chunks) in enumerate(prompt_chunks, start=1):
        # 1) Retrieved text (optionally limit number of chunks)
        lines = [f"Prompt {p_idx}: {prompt}\n"]
        for c_idx, doc in enumerate(chunks[:max_chunks], start=1) if max_chunks else enumerate(chunks, start=1):
            lines.append(f"  Chunk {c_idx}: {doc.page_content}")
        retrieved_parts.append("\n".join(lines))

        # 2) Full context block
        ctx_lines = [f"Query: {prompt}\n"]
        for c_idx, doc in enumerate(chunks, start=1):
            ctx_lines.append(f"  Chunk {c_idx}: {doc.page_content}")
        contexts.append("\n".join(ctx_lines))

    retrieved_str = "\n\n".join(retrieved_parts)
    return retrieved_str, contexts


def ingest_pipeline(
    chunk_size, chunk_overlap, embed_model, index_name, state
):
    # Load documents
    files = loader.list_filenames(index_name)
    documents = loader.load_documents(subdir=index_name, file_names=files)

    # Chunk
    chunker = DocumentChunker(
        hf_embedding_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.chunk_documents(documents)

    # Embed + index
    embeddings = HuggingFaceEmbedder(embed_model)
    vsm = VectorStoreManager(embedding_function=embeddings, index_name=index_name)
    vsm.create_index()
    vsm.add_documents(chunks)

    # Persist objects in state
    state["vsm"] = vsm
    state["embedder"] = embeddings

    # Return stats
    return str(len(files)), str(len(documents)), str(len(chunks))

def generate_pipeline(
    num_prompts, top_k, llm_model, temperature,
    max_output_tokens, sys_prompt_fuse,
    sys_prompt_final, query, state
):
    vsm = state.get("vsm")
    embeddings = state.get("embedder")
    
    final_llm = HuggingFaceLLM(model_name=llm_model)
    
    retriever = vsm.retriever(search_type = "similarity", search_kwargs = {"k":top_k})
    # 1. Generate prompts
    augmentor = PromptAugmentor(client=pa_llm)
    prompts = augmentor.generate(query=query, synthetic_count=num_prompts)

    # 2. Retrieve chunks for each prompt
    retrieved_chunk_docs = [retriever.invoke(prompt) for prompt in prompts]

    # 3. Pair prompts with their chunk lists
    prompt_chunks = list(zip(prompts, retrieved_chunk_docs))

    # 2. Format retrieved & contexts in one go
    retrieved, contexts = format_prompts_and_chunks(
        prompt_chunks,
        max_chunks=2
    )

    # 3. Generate summaries
    summaries = ""
    for idx, ctx in enumerate(contexts, start=1):
        summary = fusion_llm.get_answer(
            sys_prompt=sys_prompt_fuse,
            user_prompt=ctx,
            temperature=0.3,
            max_tokens=300
        )
        summaries += f"Prompt {idx} Summary: {summary}\n\n"

    # 4. Final answer
    final_answer = final_llm.get_answer(
        sys_prompt=sys_prompt_final,
        user_prompt=summaries,
        max_tokens=max_output_tokens,
        temperature=temperature
    )

    return prompts, retrieved, summaries, final_answer

# --- Gradio App ---
with gr.Blocks(title="Enhanced RAG App") as demo:
    gr.Markdown("## Enhanced RAG: Ingestion & Generation")
    state = gr.State({})

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
                upload_files_btn = gr.Button("Upload files")

            with gr.Column(scale=1):
                embed_model = gr.Dropdown(
                    filterable=True,
                    value="sentence-transformers/all-mpnet-base-v2",
                    choices=embedding_models,
                    label="Embedding Model"
                )
                index_name = gr.Textbox(label="Vector Index Name", value="index")
                chunk_size = gr.Number(label="Chunk Size (optional)", value=300, precision=0)
                chunk_overlap = gr.Number(label="Chunk Overlap (optional)", value=80, precision=0)
                ingest_btn = gr.Button("Run Ingestion")

        # Ensure the upload button callback is indented under the same row
        upload_files_btn.click(
            fn=upload_files,
            inputs=[file_input, index_name],
            outputs=[files_df]
        )

        with gr.Accordion(label="Ingestion Outputs", open=False):
            total_documents = gr.Textbox(label="Total Documents Processed")
            total_document_obj = gr.Textbox(label="Total Document Objects")
            total_chunks_out = gr.Textbox(label="Total Chunks Produced")

        ingest_btn.click(
            fn=ingest_pipeline,
            inputs=[chunk_size, chunk_overlap, embed_model, index_name, state],
            outputs=[total_documents, total_document_obj, total_chunks_out]
        )

    # Generation Tab
    with gr.Tab("Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                llm_model = gr.Dropdown(choices=hf_llms, label="LLM Model", value="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
                temperature = gr.Slider(0.0, 1.0, value=0.7,step=0.1, label="Temperature")
                max_output_tokens = gr.Number(label="Max Output Tokens", value=256, precision=0)
                num_prompts = gr.Number(label="Number of Prompts to Generate", value=1, precision=0)
                top_k = gr.Number(label="Top K Chunks", value=3, precision=0)

                with gr.Accordion(label="System Prompts", open=False):
                    sys_prompt_fuse = gr.Textbox(label="System Prompt for Merge Fusion", value=Prompts.MERGE_FUSION_SYS_PROMPT)
                    sys_prompt_final = gr.Textbox(label="System Prompt for Final Answer", value=Prompts.FINAL_ANS_SYS_PROMPT)

            with gr.Column(scale=1):
                user_query = gr.Textbox(label="User Query")
                generate_btn = gr.Button("Run Generation")

                with gr.Accordion(label="Intermediate Steps", open=False):
                    prompts_out = gr.Textbox(label="Augmented Prompts", lines=4)
                    retrieved_out = gr.Textbox(label="Retrieved Chunks", lines=4)
                    summaries_out = gr.Textbox(label="Summaries", lines=4)

                final_answer_out = gr.Textbox(label="Final Answer", lines=2)

        generate_btn.click(
            fn=generate_pipeline,
            inputs=[
                num_prompts, top_k, llm_model, temperature, max_output_tokens,
                sys_prompt_fuse, sys_prompt_final, user_query, state
            ],
            outputs=[prompts_out, retrieved_out, summaries_out, final_answer_out]
        )

if __name__ == "__main__":
    demo.launch()
