import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_index():
    # 1. Load documents from the office_docs folder
    loader = DirectoryLoader("office_docs/", glob="**/*.*")
    docs = loader.load()

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3. Create embeddings using a local HuggingFace model
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
    )

    # 4. Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Save index locally
    vectorstore.save_local("office_index")
    print("✅ Index built and saved to 'office_index'")


if __name__ == "__main__":
    build_index()