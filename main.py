import gradio as gr
import os

from langchain_community.vectorstores import SemaDB
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama

import ingest

embeddings = HuggingFaceEmbeddings(model_name="msmarco-MiniLM-L-6-v3", encode_kwargs={'normalize_embeddings': True})
sample_embedding = embeddings.embed_documents(["Hello World"])
vector_size = len(sample_embedding[0])

DB = SemaDB("ragchat", vector_size, embeddings, DistanceStrategy.COSINE) 
# PROMPT = """Answer the question based only on the following context. Be concise and informative.
# Context:
# {context}

# Question: {question}
# Answer:"""
PROMPT= """<|im_start|>system
You are Sema, a helpful AI assistant that answers questions based on given context.<|im_end|>
<|im_start|>user
Answer the question based only on the following context. Be concise.
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant"""

# LLM = Ollama(model="llama2")
LLM = Ollama(model="dolphin-phi")

def set_global_api_key(api_key):
    """Set the global API Key for SemaDB."""
    print("Setting API Key to", api_key)
    os.environ["SEMADB_API_KEY"] = api_key
    return "API Key Set"

def chat_response(message: str, history: list[list[str]]):
    """Chat with the model using the context from the PDFs."""
    # We are doing a simple manual RAG approach as opposed to using langchain
    # wrapper chains because we want to show the full process
    docs = DB.similarity_search(message, k=3)
    texts = [(doc.page_content, doc.metadata) for doc in docs]
    for _, meta in texts:
        del meta["text"]
    # ctx = "\n\n".join([f"(Metadata {meta}) {txt}" for txt, meta in texts])
    ctx = "\n\n".join([txt for txt, _ in texts])
    prompt = PROMPT.format(context=ctx, question=message)
    print("Prompt\n", prompt)
    # ---------------------------
    response = ""
    for chunk in LLM.stream(prompt):
        response += chunk
        yield response

def process_pdfs(filepaths: list[str], progress=gr.Progress()):
    """Process the PDFs and add them to the database."""
    print("Loading", filepaths)
    texts = []
    metadatas = []
    for fpath in progress.tqdm(filepaths):
        docs = ingest.load_pdf(fpath)
        for doc in docs:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
    print("Adding texts to database", len(texts))
    # ---------------------------
    # Clear existing collection
    print("Creating collection")
    if not DB.create_collection():
        print("Error creating collection, trying to delete and recreate...")
        if not DB.delete_collection():
            return "Error deleting existing collection"
        if not DB.create_collection():
            return "Error creating collection"
    print("Collection created")
    # ---------------------------
    ids = DB.add_texts(texts, metadatas)
    if len(ids) != len(docs):
        return "Error adding texts to database"
    return f"Processed PDFs - Chunk Count: {len(texts)}"

def process_wikipedia(search: str):
    """Process the Wikipedia page and add it to the database."""
    print("Loading Wikipedia", search)
    docs = ingest.load_wikipedia(search)
    texts = [doc.page_content for doc in docs]
    print("Adding texts to database", len(texts))
    # ---------------------------
    # Clear existing collection
    print("Creating collection")
    if not DB.create_collection():
        print("Error creating collection, trying to delete and recreate...")
        if not DB.delete_collection():
            return "Error deleting existing collection"
        if not DB.create_collection():
            return "Error creating collection"
    print("Collection created")
    # ---------------------------
    ids = DB.add_texts(texts)
    if len(ids) != len(docs):
        return "Error adding texts to database"
    return f"Processed Wikipedia - Chunk Count: {len(texts)}"

# ---------------------------

def main():
    with gr.Blocks() as demo:
        # ---------------------------
        # api_key = gr.Textbox(label="SemaDB API Key")
        # set_api_key = gr.Button("Set API Key")
        # set_api_key.click(set_global_api_key, api_key, api_key)
        # ---------------------------
        with gr.Row():
            fupload = gr.File(label="Upload PDF", file_count="multiple", file_types=[".pdf"])
            process_pdf_output  = gr.Textbox(label="Process output")
            fupload.upload(process_pdfs, fupload, process_pdf_output)
        # ---------------------------
        wikipedia_search = gr.Textbox(label="Search Wikipedia")
        wikipedia_search_output = gr.Textbox(label="Wikipedia Output")
        wikipedia_search_button = gr.Button("Search Wikipedia")
        wikipedia_search_button.click(process_wikipedia, wikipedia_search, wikipedia_search_output)
        # ---------------------------
        gr.ChatInterface(chat_response,
                        title="RAG Chat",
                        description="Chat with the model using the retrieval augmented generation model.",
                        fill_height=True)

    demo.launch()

if __name__ == "__main__":
    main()