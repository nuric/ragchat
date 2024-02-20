from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pdfminer.high_level import extract_text

import wikipedia

text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 512,
chunk_overlap  = 32,
length_function = len,
)

def load_pdf(pdf_path: str) -> list[Document]:
    """Load a PDF and split it into chunks."""
    full_text = extract_text(pdf_path)
    print("PDF Content", pdf_path, full_text)
    chunks = text_splitter.split_text(full_text)
    chunks = [" ".join(chunk.split()) for chunk in chunks if chunk.strip()]
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

def load_wikipedia(search: str) -> list[Document]:
    """Load a wikipedia page and split it into chunks."""
    pages = wikipedia.search(search)
    if not pages:
        return []
    page = wikipedia.page(pages[0])
    text = page.content
    # metadata = {
    #     "title": page.title,
    #     "url": page.url,
    # }
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

if __name__ == "__main__":
    res = load_pdf("demo.pdf")
    print(res)
    print(len(res))
    res = load_wikipedia("Imperial College")
    print(res)
    print(len(res))