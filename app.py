# new_gradio_app.py

import gradio as gr
import time
import random
from src.generation.Prompts import Prompts

# Dummy LLM models and embedding models (for dropdowns)
hf_llms = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
]

embedding_models = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "hkunlp/instructor-large"
]

# Dummy knowledge graph endpoints
knowledge_graph_endpoints = [
    "Neo4j Instance 1 (Production)",
    "Neo4j Instance 2 (Staging)",
    "Neo4j Localhost (Development)"
]

# --- Dummy Streaming Generation for Full Hybrid RAG ---
def streaming_generate_full(num_prompts: int, top_k: int, llm_model: str, temperature: float,
                            max_output_tokens: int, sys_prompt_fuse: str,
                            sys_prompt_final: str, show_chunks: int, query: str, kg_endpoint: str):
    start_time = time.time()

    # Hardcoded Augmented Prompts
    hardcoded_prompts = [
        "Define a knowledge graph database, detailing its core components and architectural principles.",
        "Explain how knowledge graph databases differ from traditional relational or NoSQL databases.",
        "What are the primary use cases and advantages of implementing a knowledge graph database?",
        "Describe Neo4j's unique features and how it implements the property graph model.",
        "Compare Neo4j's query language (Cypher) with other graph query languages or methods used by competitors.",
        "Analyze Neo4j's performance characteristics, scalability, and security features in the context of enterprise deployments.",
        "Identify Neo4j's main competitors in the knowledge graph database market and their respective strengths and weaknesses.",
        "Summarize the key differentiators and competitive advantages that Neo4j offers over alternative graph database solutions.",
    ]
    prompts_out = "\n\n".join(hardcoded_prompts[:num_prompts]) # Display only up to num_prompts
    yield (prompts_out, "", "", "", "", "Calculating...", "Calculating...")
    time.sleep(random.uniform(9, 17)) # Simulate processing time

    # Hardcoded Retrieved Chunks
    hardcoded_chunks = [
        ["A knowledge graph database fundamentally represents data as a network of interconnected entities (nodes) and relationships (edges), storing semantic information about how these entities relate.",
         "Key components include nodes for entities, relationships for connections, and properties to describe both, enabling a highly intuitive and flexible data model."],
        ["Unlike relational databases that rely on rigid schemas and joins, knowledge graphs excel at handling complex, evolving relationships, allowing for flexible data exploration without predefined table structures.",
         "Compared to typical NoSQL document or key-value stores, graph databases explicitly model relationships as first-class citizens, making traversal and pattern matching highly efficient."],
        ["Primary use cases include fraud detection, recommendation engines, master data management, and contextual search, where understanding complex relationships is critical.",
         "Advantages include enhanced data interconnectedness, agile schema evolution, faster complex query execution, and improved AI/ML model training due to richer context."],
        ["Neo4j is a native graph database implementing the labeled property graph model, where nodes and relationships can have labels and properties, enabling highly expressive data representation.",
         "Its architecture is optimized for graph traversals, ensuring high performance even on deeply connected datasets, a hallmark of its design philosophy."],
        ["Cypher, Neo4j's declarative query language, is known for its readability and expressive power, allowing users to describe visual graph patterns in a highly intuitive syntax.",
         "Competitors might use SPARQL (for RDF graphs), Gremlin (for TinkerPop-compatible graphs), or SQL extensions, each with varying levels of expressiveness and learning curves compared to Cypher."],
        ["Neo4j boasts strong performance for connected data queries, often outperforming other database types by orders of magnitude for pathfinding and relationship-intensive operations.",
         "It offers robust clustering and replication for scalability and high availability, alongside enterprise-grade security features like role-based access control and encryption."],
        ["Key competitors include Amazon Neptune (supporting Gremlin and SPARQL), ArangoDB (multi-model with graph capabilities), and Microsoft Azure Cosmos DB (with a Gremlin API).",
         "Each competitor has different strengths, such as integration with specific cloud ecosystems, multi-model flexibility, or serverless deployment options."],
        ["Neo4j's deep focus on the property graph model, combined with the mature Cypher language and a vibrant developer ecosystem, provides a compelling offering.",
         "Its strong community support, extensive documentation, and proven track record in complex enterprise solutions often set it apart in the dedicated graph database space."]
    ]

    retrieved_out = ""
    for i in range(min(num_prompts, len(hardcoded_chunks))): # Iterate up to num_prompts or available chunks
        retrieved_out += f"Prompt {i+1} → Chunks:\n"
        for j in range(min(show_chunks, len(hardcoded_chunks[i]))): # Display up to show_chunks
            retrieved_out += f"- {hardcoded_chunks[i][j]}\n"
        retrieved_out += "\n"
    yield (prompts_out, retrieved_out, "", "", "", "Calculating...", "Calculating...")

    time.sleep(random.uniform(15, 30)) # Simulate processing time

    # Hardcoded Summaries
    hardcoded_summaries = [
        "A knowledge graph database defines data by nodes and relationships, with properties, enabling semantic connections.",
        "Unlike traditional databases, knowledge graphs prioritize relationships for flexible, complex data navigation.",
        "They are crucial for use cases like fraud detection and offer advantages in data interconnectedness and agile schemas.",
        "Neo4j uses a native property graph model, optimizing its architecture for efficient graph traversals.",
        "Cypher is Neo4j's intuitive query language, differing from SPARQL or Gremlin used by other graph systems.",
        "Neo4j is performance-optimized for connected data, offering robust scalability and enterprise security features.",
        "Competitors like Amazon Neptune and ArangoDB offer different features, often tied to cloud platforms or multi-model support.",
        "Neo4j distinguishes itself with its dedicated property graph focus, Cypher, and strong ecosystem for complex enterprise needs."
    ]
    summaries_out = "\n\n".join(hardcoded_summaries[:num_prompts]) # Display only up to num_prompts
    yield (prompts_out, retrieved_out, summaries_out, "", "", "Calculating...", "Calculating...")

    time.sleep(random.uniform(1, 4)) # Simulate processing time

    # Hardcoded Knowledge Graph Response
    kg_response_out = f"""Knowledge Graph Response for '{query}' from {kg_endpoint}:
Entities found: Knowledge Graph Database, Neo4j, Property Graph Model, Cypher, Relational Databases, NoSQL Databases, Amazon Neptune, ArangoDB.
Relationships: `Knowledge Graph Database` --[HAS_CHARACTERISTIC]-> `Nodes`, `Knowledge Graph Database` --[HAS_CHARACTERISTIC]-> `Relationships`, `Neo4j` --[IMPLEMENTS]-> `Property Graph Model`, `Neo4j` --[USES_QUERY_LANGUAGE]-> `Cypher`, `Knowledge Graph Database` --[DIFFER_FROM]-> `Relational Databases`, `Knowledge Graph Database` --[DIFFER_FROM]-> `NoSQL Databases`, `Neo4j` --[COMPETES_WITH]-> `Amazon Neptune`, `Neo4j` --[COMPETES_WITH]-> `ArangoDB`.
Relevant facts: Knowledge graphs emphasize relationships as first-class citizens. Neo4j's Cypher is a declarative graph query language. Key benefits of KGs include improved contextual search and recommendation. Neo4j is known for its traversal performance."""
    yield (prompts_out, retrieved_out, summaries_out, kg_response_out, "", "Calculating...", "Calculating...")

    time.sleep(random.uniform(5, 14)) # Simulate processing time

    # Hardcoded Final Answer
    final_out = f"""A **knowledge graph database** is a specialized type of database that stores data as a network of interconnected entities (nodes) and their relationships (edges), enriched with properties. This model intuitively represents complex, real-world data and its underlying semantics, making it highly effective for applications requiring deep contextual understanding and relationship analysis, such as fraud detection, recommendation engines, and master data management.

**Neo4j** stands out among its competitors primarily due to its unwavering focus on the **property graph model** as its native data structure, which provides superior performance for connected data traversals. Its intuitive and declarative query language, **Cypher**, allows users to express complex graph patterns with remarkable clarity, simplifying development. While competitors like Amazon Neptune (which supports Gremlin and SPARQL) and multi-model databases like ArangoDB offer various strengths, Neo4j's maturity, robust ecosystem, strong community support, and enterprise-grade features (including scalability, high availability, and security) often position it as a leader for dedicated, high-performance graph database solutions. Its commitment to the graph paradigm, coupled with extensive tooling and a proven track record, gives Neo4j a significant competitive advantage."""
    yield (prompts_out, retrieved_out, summaries_out, kg_response_out, final_out, "Calculating...", "Calculating...")

    # Calculate LLM Calls and Time Taken
    total_llm_calls = (3 * num_prompts) + 1
    time_taken = f"{time.time() - start_time:.2f} seconds"

    yield (prompts_out, retrieved_out, summaries_out, kg_response_out, final_out, str(total_llm_calls), time_taken)


# --- Dummy Streaming Generation for Fast Hybrid RAG ---
def streaming_generate_fast(num_prompts: int, top_k: int, llm_model: str, temperature: float,
                            max_output_tokens: int, sys_prompt_fuse: str,
                            sys_prompt_final: str, show_chunks: int, query: str, kg_endpoint: str):
    start_time = time.time()

    # Hardcoded Augmented Prompts
    hardcoded_prompts = [
        "Define a knowledge graph database, detailing its core components and architectural principles.",
        "Explain how knowledge graph databases differ from traditional relational or NoSQL databases.",
        "What are the primary use cases and advantages of implementing a knowledge graph database?",
        "Describe Neo4j's unique features and how it implements the property graph model.",
        "Compare Neo4j's query language (Cypher) with other graph query languages or methods used by competitors.",
        "Analyze Neo4j's performance characteristics, scalability, and security features in the context of enterprise deployments.",
        "Identify Neo4j's main competitors in the knowledge graph database market and their respective strengths and weaknesses.",
        "Summarize the key differentiators and competitive advantages that Neo4j offers over alternative graph database solutions.",
    ]
    prompts_out = "\n\n".join(hardcoded_prompts[:num_prompts])
    yield (prompts_out, "", "", "Calculating...", "Calculating...")
    time.sleep(random.uniform(2, 4)) # Simulate processing time

    # Hardcoded Retrieved Chunks (includes KG response for Fast RAG)
    hardcoded_chunks = [
        ["A knowledge graph database fundamentally represents data as a network of interconnected entities (nodes) and relationships (edges), storing semantic information about how these entities relate.",
         "Key components include nodes for entities, relationships for connections, and properties to describe both, enabling a highly intuitive and flexible data model."],
        ["Unlike relational databases that rely on rigid schemas and joins, knowledge graphs excel at handling complex, evolving relationships, allowing for flexible data exploration without predefined table structures.",
         "Compared to typical NoSQL document or key-value stores, graph databases explicitly model relationships as first-class citizens, making traversal and pattern matching highly efficient."],
        ["Primary use cases include fraud detection, recommendation engines, master data management, and contextual search, where understanding complex relationships is critical.",
         "Advantages include enhanced data interconnectedness, agile schema evolution, faster complex query execution, and improved AI/ML model training due to richer context."],
        ["Neo4j is a native graph database implementing the labeled property graph model, where nodes and relationships can have labels and properties, enabling highly expressive data representation.",
         "Its architecture is optimized for graph traversals, ensuring high performance even on deeply connected datasets, a hallmark of its design philosophy."],
        ["Cypher, Neo4j's declarative query language, is known for its readability and expressive power, allowing users to describe visual graph patterns in a highly intuitive syntax.",
         "Competitors might use SPARQL (for RDF graphs), Gremlin (for TinkerPop-compatible graphs), or SQL extensions, each with varying levels of expressiveness and learning curves compared to Cypher."],
        ["Neo4j boasts strong performance for connected data queries, often outperforming other database types by orders of magnitude for pathfinding and relationship-intensive operations.",
         "It offers robust clustering and replication for scalability and high availability, alongside enterprise-grade security features like role-based access control and encryption."],
        ["Key competitors include Amazon Neptune (supporting Gremlin and SPARQL), ArangoDB (multi-model with graph capabilities), and Microsoft Azure Cosmos DB (with a Gremlin API).",
         "Each competitor has different strengths, such as integration with specific cloud ecosystems, multi-model flexibility, or serverless deployment options."],
        ["Neo4j's deep focus on the property graph model, combined with the mature Cypher language and a vibrant developer ecosystem, provides a compelling offering.",
         "Its strong community support, extensive documentation, and proven track record in complex enterprise solutions often set it apart in the dedicated graph database space."]
    ]

    retrieved_out = ""
    for i in range(min(num_prompts, len(hardcoded_chunks))):
        retrieved_out += f"Prompt {i+1} → Chunks:\n"
        for j in range(min(show_chunks, len(hardcoded_chunks[i]))):
            retrieved_out += f"- {hardcoded_chunks[i][j]}\n"
        retrieved_out += "\n"

    # Hardcoded Knowledge Graph Response (integrated into retrieved chunks for Fast RAG)
    kg_response_in_chunks = f"""Knowledge Graph Response for '{query}' from {kg_endpoint} (Fast RAG):
Entities found: Knowledge Graph, Neo4j, Cypher, Graph Databases.
Relevant connections: `Knowledge Graph` --[DEFINES]-> `Relationships`, `Neo4j` --[FEATURES]-> `Cypher`, `Neo4j` --[COMPETES_IN]-> `Graph Databases Market`.
Summary of graph data for all prompts: Knowledge graphs model interconnected data; Neo4j is a leading graph database with its native property graph and Cypher query language, offering competitive advantages in performance for connected data. Key competitors exist, but Neo4j maintains strong market position through focused innovation and ecosystem."""
    retrieved_out += f"\n---\nKnowledge Graph Integration:\n{kg_response_in_chunks}\n---"
    yield (prompts_out, retrieved_out, "", "Calculating...", "Calculating...")
    time.sleep(random.uniform(1, 3)) # Simulate processing time

    # Hardcoded Final Answer
    final_out = f"""For '{query}', the **Fast Hybrid RAG** approach quickly synthesizes insights by directly integrating relevant knowledge graph data with retrieved document chunks. This streamlined process prioritizes efficiency, delivering a concise yet comprehensive answer focused on the most pertinent information about knowledge graph databases and Neo4j's competitive advantages."""
    yield (prompts_out, retrieved_out, final_out, "Calculating...", "Calculating...")

    # Calculate LLM Calls and Time Taken
    total_llm_calls = num_prompts + 2
    time_taken = f"{time.time() - start_time:.2f} seconds" # Using actual time difference

    yield (prompts_out, retrieved_out, final_out, str(total_llm_calls), time_taken)


# --- Build UI ---
with gr.Blocks(title="Hybrid RAG App") as demo:
    gr.Markdown("## Hybrid RAG: Full Hybrid RAG vs. Fast Hybrid RAG")

    with gr.Tab("Full Hybrid RAG"):
        with gr.Row():
            with gr.Column(scale=1):
                llm_model_full = gr.Dropdown(label="LLM Model", choices=hf_llms, value=hf_llms[-1])
                temperature_full = gr.Slider(0.0, 1.0, value=0.7, step=0.1)

                with gr.Row():
                    max_output_tokens_full = gr.Number(label="Max Output Tokens", value=256, precision=0)
                    # Adjusted default for num_prompts to 8 for the example
                    num_prompts_full = gr.Number(label="Number of Prompts", value=8, precision=0)

                with gr.Row():
                    # Adjusted default for top_k to 2 to match chunk output
                    top_k_full = gr.Number(label="Top-K Chunks", value=2, precision=0)
                    show_chunks_full = gr.Number(label="Chunks to Display", value=2, precision=0)

                kg_endpoint_full = gr.Dropdown(label="Knowledge Graph Endpoint", choices=knowledge_graph_endpoints, value=knowledge_graph_endpoints[0])

                with gr.Accordion("System Prompts", open=False):
                    sys_fuse_full = gr.Textbox(label="Merge Fusion Prompt",
                                          value=Prompts.MERGE_FUSION_SYS_PROMPT, lines=4)
                    sys_final_full = gr.Textbox(label="Final Answer Prompt",
                                           value=Prompts.FINAL_ANS_SYS_PROMPT, lines=4)

            with gr.Column(scale=1):
                # Hardcoded user query
                user_query_full = gr.Textbox(label="User Query", value="What is a knowledge graph database, and what does Neo4j bring to the table as compared to competitors?")
                run_btn_full = gr.Button("Run Full Hybrid RAG")

                final_out_full = gr.Markdown(label="Final Answer", show_label=True)

                with gr.Accordion("Intermediate Steps", open=False):
                    prompts_out_full = gr.Textbox(label="Augmented Prompts", lines=4)
                    retrieved_out_full = gr.Textbox(label="Retrieved Chunks", lines=4)
                    summaries_out_full = gr.Textbox(label="Summaries", lines=4)
                    kg_response_out_full = gr.Textbox(label="Knowledge Graph Response", lines=4)

                with gr.Row():
                    total_llm_calls_full = gr.Textbox(label="Total LLM Calls ((3n+1) where n = num_prompts)", interactive=False)
                    time_taken_full = gr.Textbox(label="Time Taken to Generate", interactive=False)


        run_btn_full.click(
            fn=streaming_generate_full,
            inputs=[num_prompts_full, top_k_full, llm_model_full, temperature_full,
                    max_output_tokens_full, sys_fuse_full, sys_final_full,
                    show_chunks_full, user_query_full, kg_endpoint_full],
            outputs=[prompts_out_full, retrieved_out_full, summaries_out_full, kg_response_out_full,
                     final_out_full, total_llm_calls_full, time_taken_full]
        )

    with gr.Tab("Fast Hybrid RAG"):
        with gr.Row():
            with gr.Column(scale=1):
                llm_model_fast = gr.Dropdown(label="LLM Model", choices=hf_llms, value=hf_llms[-1])
                temperature_fast = gr.Slider(0.0, 1.0, value=0.7, step=0.1)

                with gr.Row():
                    max_output_tokens_fast = gr.Number(label="Max Output Tokens", value=256, precision=0)
                    # Adjusted default for num_prompts to 8 for the example
                    num_prompts_fast = gr.Number(label="Number of Prompts", value=8, precision=0)

                with gr.Row():
                    top_k_fast = gr.Number(label="Top-K Chunks", value=3, precision=0)
                    show_chunks_fast = gr.Number(label="Chunks to Display", value=2, precision=0)

                kg_endpoint_fast = gr.Dropdown(label="Knowledge Graph Endpoint", choices=knowledge_graph_endpoints, value=knowledge_graph_endpoints[2]) # Default to localhost for fast RAG

                with gr.Accordion("System Prompts", open=False):
                    sys_fuse_fast = gr.Textbox(label="Merge Fusion Prompt",
                                          value=Prompts.MERGE_FUSION_SYS_PROMPT, lines=4)
                    sys_final_fast = gr.Textbox(label="Final Answer Prompt",
                                           value=Prompts.FINAL_ANS_SYS_PROMPT, lines=4)

            with gr.Column(scale=1):
                # Hardcoded user query
                user_query_fast = gr.Textbox(label="User Query", value="What is a knowledge graph database, and what does Neo4j bring to the table as compared to competitors?")
                run_btn_fast = gr.Button("Run Fast Hybrid RAG")

                final_out_fast = gr.Markdown(label="Final Answer", show_label=True)

                with gr.Accordion("Intermediate Steps", open=False):
                    prompts_out_fast = gr.Textbox(label="Augmented Prompts", lines=4)
                    retrieved_out_fast = gr.Textbox(label="Retrieved Chunks (includes KG response)", lines=4) # Changed label

                with gr.Row():
                    total_llm_calls_fast = gr.Textbox(label="Total LLM Calls ((n+2) where n = num_prompts)", interactive=False)
                    time_taken_fast = gr.Textbox(label="Time Taken to Generate", interactive=False)

        run_btn_fast.click(
            fn=streaming_generate_fast,
            inputs=[num_prompts_fast, top_k_fast, llm_model_fast, temperature_fast,
                    max_output_tokens_fast, sys_fuse_fast, sys_final_fast,
                    show_chunks_fast, user_query_fast, kg_endpoint_fast],
            outputs=[prompts_out_fast, retrieved_out_fast, final_out_fast,
                     total_llm_calls_fast, time_taken_fast]
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch()