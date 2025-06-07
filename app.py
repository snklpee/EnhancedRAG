# app.py

import os
import shutil
import gradio as gr
from typing import Dict, List

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

# ——— Singletons & globals ———
pa_llm     = HuggingFaceLLM(model_name="meta-llama/Llama-3.1-8B-Instruct")
fusion_llm = HuggingFaceLLM(model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
_final_llm_cache: Dict[str, HuggingFaceLLM] = {}

loader    = DocumentLoader()
augmentor = PromptAugmentor(client=pa_llm)

def get_final_llm(model_name: str) -> HuggingFaceLLM:
    if model_name not in _final_llm_cache:
        _final_llm_cache[model_name] = HuggingFaceLLM(model_name=model_name)
    return _final_llm_cache[model_name]

# ——— Streaming Ingestion ———
def streaming_ingest(chunk_size, chunk_overlap, embed_model, index_name, ingest_state):
    # 0% → list files
    files = loader.list_filenames(index_name)
    if not files:
        raise gr.Error(f"No files under index `{index_name}`.")
    yield (str(len(files)), "", "")

    # 25% → load docs
    docs = loader.load_documents(subdir=index_name, file_names=files)
    yield (str(len(files)), str(len(docs)), "")

    # 50% → chunk documents
    chunker = DocumentChunker(
        hf_embedding_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.chunk_documents(docs)
    yield (str(len(files)), str(len(docs)), str(len(chunks)))

    # 75% → build index
    embeddings = HuggingFaceEmbedder(embed_model)
    vsm = VectorStoreManager(embedding_function=embeddings, index_name=index_name)
    vsm.create_index()
    vsm.add_documents(chunks)
    yield (str(len(files)), str(len(docs)), str(len(chunks)))

    # 100% → done
    ingest_state["vsm"]       = vsm
    ingest_state["embedder"] = embeddings
    yield (str(len(files)), str(len(docs)), str(len(chunks)))

# ——— Streaming Generation ———
def streaming_generate(num_prompts: int, top_k: int, llm_model: str, temperature: float,
                       max_output_tokens: int, sys_prompt_fuse: str,
                       sys_prompt_final: str, show_chunks: int, query: str, ingest_state):
    vsm = ingest_state.get("vsm")
    if vsm is None:
        raise gr.Error("Run ingestion first.")

    # 1) generate prompts
    prompts = augmentor.generate(query=query, synthetic_count=num_prompts)
    prompts_out = "\n\n".join(prompts)
    yield (prompts_out, "", "", "")

    # 2) retrieve chunks
    retriever = vsm.retriever(search_type="similarity", search_kwargs={"k": top_k})
    retrieved_out = ""
    prompt_chunks = []
    for idx, p in enumerate(prompts, start=1):
        docs = retriever.invoke(p)
        prompt_chunks.append((p, docs))
        snippet = "\n".join(docs[:show_chunks])
        retrieved_out += f"Prompt {idx} → Chunks:\n{snippet}\n\n"
        yield (prompts_out, retrieved_out, "", "")

    # 3) intermediate summaries
    summarizer = FusionSummarizer(fusion_llm=fusion_llm, sys_prompt=sys_prompt_fuse)
    summaries = summarizer.summarize(prompt_chunks=prompt_chunks)
    summaries_out = "\n\n".join(summaries)
    yield (prompts_out, retrieved_out, summaries_out, "")

    # 4) final answer
    final_llm = get_final_llm(llm_model)
    final_out = final_llm.get_answer(
        sys_prompt=sys_prompt_final,
        user_prompt=summaries_out,
        max_tokens=max_output_tokens,
        temperature=temperature
    )
    yield (prompts_out, retrieved_out, summaries_out, final_out)

# ——— Build UI ———
lister          = HuggingFaceModelLister()
embedding_models = lister.list_models(task="sentence-similarity", filter="feature-extraction")
hf_llms         = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
]

with gr.Blocks(title="Enhanced RAG App") as demo:
    session_state = gr.State({})

    gr.Markdown("## Enhanced RAG: Ingestion & Generation")

    # Ingestion Tab
    with gr.Tab("Ingestion"):
        with gr.Row():
            with gr.Column(scale=3):
                file_input   = gr.File(label="Upload Documents", file_count="multiple",
                                       file_types=[".pdf", ".txt", ".py", ".json", ".rs", ".cpp"])
                files_df     = gr.Dataframe(headers=["Sr. No", "File Name"],
                                            datatype=["number", "str"],
                                            interactive=False)
                
                with gr.Row():
                    upload_btn   = gr.Button("Upload Files")
                    cleanup_btn  = gr.Button("Clear Files")
                with gr.Row():
                    upload_status  = gr.Textbox(label="Status", interactive=False)
                    cleanup_status = gr.Textbox(label="Cleanup Status", interactive=False)

            with gr.Column(scale=2):
                embed_model  = gr.Dropdown(label="Embedding Model",
                                          choices=embedding_models,
                                          value=embedding_models[0])
                with gr.Row():
                    chunk_size   = gr.Number(label="Chunk Size", value=300, precision=0)
                    chunk_overlap= gr.Number(label="Chunk Overlap", value=80, precision=0)
                    
                index_name   = gr.Textbox(label="Vector Index Name", value="index")
                ingest_btn   = gr.Button("Run Ingestion")
                
        ingestion_status = gr.Textbox(label="Document Ingestion Status")
                
        with gr.Accordion(label="Stats", open=False):
            total_files  = gr.Textbox(label="Total Files")
            total_docs   = gr.Textbox(label="Total Documents")
            total_chunks = gr.Textbox(label="Total Chunks")
                
        
            
        upload_btn.click(
            fn=lambda files, name: (
                [[i+1, os.path.basename(f.name)] for i, f in enumerate(files)],
                "Uploaded"
            ),
            inputs=[file_input, index_name],
            outputs=[files_df, upload_status]
        )

        cleanup_btn.click(
            fn=lambda name: (shutil.rmtree(os.path.join(settings.CONTEXT_DIR, name), ignore_errors=True), "Cleared")[1],
            inputs=index_name,
            outputs=cleanup_status
        )

        ingest_btn.click(
            fn=streaming_ingest,
            inputs=[chunk_size, chunk_overlap, embed_model, index_name, session_state],
            outputs=[total_files, total_docs, total_chunks]
        )

    # Generation Tab
    with gr.Tab("Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                llm_model = gr.Dropdown(label="LLM Model", choices=hf_llms, value=hf_llms[-1])
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1)
                
                with gr.Row():
                    max_output_tokens = gr.Number(label="Max Output Tokens", value=256, precision=0)
                    num_prompts = gr.Number(label="Number of Prompts", value=1, precision=0)
                    
                with gr.Row():
                    top_k = gr.Number(label="Top-K Chunks", value=3, precision=0)
                    show_chunks = gr.Number(label="Chunks to Display", value=2, precision=0)
                    
                with gr.Accordion("System Prompts", open=False):
                    sys_fuse = gr.Textbox(label="Merge Fusion Prompt",
                                          value=Prompts.MERGE_FUSION_SYS_PROMPT, lines=4)
                    sys_final = gr.Textbox(label="Final Answer Prompt",
                                           value=Prompts.FINAL_ANS_SYS_PROMPT, lines=4)

            with gr.Column(scale=1):
                user_query = gr.Textbox(label="User Query")
                run_btn = gr.Button("Run Generation")
                
                final_out = gr.Markdown(label="Final Answer", show_label=True)

                with gr.Accordion("Intermediate Steps", open=False):
                    prompts_out = gr.Textbox(label="Augmented Prompts", lines=4)
                    retrieved_out = gr.Textbox(label="Retrieved Chunks", lines=4)
                    summaries_out = gr.Textbox(label="Summaries", lines=4)
                    

        run_btn.click(
            fn=streaming_generate,
            inputs=[num_prompts, top_k, llm_model, temperature,
                    max_output_tokens, sys_fuse, sys_final,
                    show_chunks, user_query, session_state],
            outputs=[prompts_out, retrieved_out, summaries_out, final_out]
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
