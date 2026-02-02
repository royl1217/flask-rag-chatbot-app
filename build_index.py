import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_documents():
    docs = []
    for file in os.listdir("office_docs"):
        if file.endswith(".pdf"):
            path = os.path.join("office_docs", file)
            try:
                loader = PyPDFLoader(path)
                pdf_docs = loader.load()
                if len(pdf_docs) == 0:
                    raise ValueError("Empty PDF")
                docs.extend(pdf_docs)
            except:
                print(f"⚠️ Falling back to OCR for {file}")
                loader = UnstructuredPDFLoader(path)
                docs.extend(loader.load())
    return docs

def build_index():
    docs = load_documents()
    print("Docs loaded:", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    print("Chunks:", len(chunks))

    if not chunks:
        print("❌ No valid text found. Cannot build index.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Generating embeddings...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Saving index...")
    vectorstore.save_local("office_index")
    print("✅ Index saved to office_index/")

if __name__ == "__main__":
    build_index()