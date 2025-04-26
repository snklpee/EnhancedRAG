# app.py

import os
import shutil
import gradio as gr

import pandas as pd

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.HuggingFaceEmbedder import HuggingFaceEmbedder
from src.ingestion.VectorStoreManager import VectorStoreManager 
from src.utils.ModelLister import HuggingFaceModelLister

from config.settings import settings

# Model lists
lister = HuggingFaceModelLister()
embedding_models = lister.list_models(task="sentence-similarity", filter="feature-extraction")
llms = lister.list_models(task="text-generation", filter="text-generation-inference")

# A single loader can be reused across runs
loader = DocumentLoader()

print(settings.CONTEXT_DIR)

def upload_files(uploaded_files,index_name):
    save_dir = os.path.join(settings.CONTEXT_DIR,index_name)
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for f in uploaded_files:
        dest = os.path.join(save_dir, f.orig_name or os.path.basename(f.name))
        shutil.copyfile(f.name, dest)
        saved_paths.append(dest)
    files = [[i+1, os.path.basename(p)] for i, p in enumerate(saved_paths)]
    df = pd.DataFrame({
        "Sr. No":range(1,len(files)+1), 
        "File Name":files})
    return df
        

def ingest_pipeline(
    chunk_size, chunk_overlap, embed_model, index_name,
    state  # this is a dict passed in via gr.State
):
    # 2. Load documents
    files = loader.list_filenames(index_name)
    documents = loader.load_documents(subdir=index_name, file_names=files)

    # 3. Chunk
    chunker = DocumentChunker(
        hf_embedding_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.chunk_documents(documents)

    # 4. Embed + index
    embeddings = HuggingFaceEmbedder(embed_model)
    vsm = VectorStoreManager(embedding_function=embeddings, index_name=index_name)
    vsm.create_index()
    vsm.add_documents(chunks)

    # 5. Persist objects in state
    state["chunker"] = chunker
    state["vsm"] = vsm
    state["embedder"] = embeddings

    # Return ingestion stats + the saved paths for DataFrame
    return (
        str(len(files)),
        str(len(documents)),
        str(len(chunks)),
        )

def generate_pipeline(
    num_prompts, top_k, llm_model, temperature,
    max_output_tokens, sys_prompt_aug, sys_prompt_fuse,
    sys_prompt_final, query, state
):
    # Retrieve persisted objects
    chunker: DocumentChunker = state.get("chunker")
    vsm: VectorStoreManager = state.get("vsm")
    embeddings: HuggingFaceEmbedder = state.get("embedder")

    # TODO: use chunker, vsm, embeddings for real logic
    prompts = [f"[Augmented Prompt {i+1}]" for i in range(int(num_prompts))]
    retrieved = [["Chunk1", "Chunk2"] for _ in prompts]
    summaries = [f"[Summary {i+1}]" for _ in prompts]
    final_answer = "[Final synthesized answer]"

    return prompts, retrieved, summaries, final_answer

# --- Gradio App ---
with gr.Blocks(title="Enhanced RAG App") as demo:
    gr.Markdown("## Enhanced RAG: Ingestion & Generation")

    # A State component to hold our pipeline objects
    state = gr.State({})

    # ----------------- Ingestion Tab -----------------
    with gr.Tab("Ingestion"):
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".py", ".json"]
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
                index_name = gr.Textbox(label="Vector Index Name")
                chunk_size = gr.Number(label="Chunk Size (optional)", value=None, precision=0)
                chunk_overlap = gr.Number(label="Chunk Overlap (optional)", value=None, precision=0)
                ingest_btn = gr.Button("Run Ingestion")

        upload_files_btn.click(
        fn=upload_files,
        inputs=[file_input, index_name],
        outputs=[files_df])

        with gr.Accordion(label="Ingestion Outputs", open=False):
            total_documents = gr.Textbox(label="Total Documents Processed")
            total_document_obj = gr.Textbox(label="Total Document Objects")
            total_chunks_out = gr.Textbox(label="Total Chunks Produced")
        

        ingest_btn.click(
            fn=ingest_pipeline,
            inputs=[chunk_size, chunk_overlap, embed_model, index_name, state],
            outputs=[total_documents, total_document_obj, total_chunks_out]
        )

    # ----------------- Generation Tab -----------------
    with gr.Tab("Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                llm_model = gr.Dropdown(choices=llms, label="LLM Model")
                temperature = gr.Slider(0.0, 1.0, value=0.7, label="Temperature")
                max_output_tokens = gr.Number(label="Max Output Tokens", value=256, precision=0)
                num_prompts = gr.Number(label="Number of Prompts to Generate", value=1, precision=0)
                top_k = gr.Number(label="Top K Chunks", value=3, precision=0)

                with gr.Accordion(label="System Prompts", open=False):
                    sys_prompt_aug = gr.Textbox(label="System Prompt for Prompt Augmentation")
                    sys_prompt_fuse = gr.Textbox(label="System Prompt for Merge Fusion")
                    sys_prompt_final = gr.Textbox(label="System Prompt for Final Answer")

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
                sys_prompt_aug, sys_prompt_fuse, sys_prompt_final, user_query, state
            ],
            outputs=[prompts_out, retrieved_out, summaries_out, final_answer_out]
        )

if __name__ == "__main__":
    demo.launch()
