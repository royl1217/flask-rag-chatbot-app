from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

# Load FAISS index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    vectorstore = FAISS.load_local(
        "office_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# DeepInfra (OpenAI-compatible) LLM
llm = ChatOpenAI(
    model=os.getenv("DEEPINFRA_MODEL", "google/gemma-2-9b-it"),
    openai_api_key=os.getenv("DEEPINFRA_API_KEY"),
    openai_api_base="https://api.deepinfra.com/v1/openai",
    temperature=0.2,
    max_tokens=256
)



# Prompt template
template = """
You are an assistant answering questions based on the provided context.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# RAG chain
rag_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = rag_chain.invoke(question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)