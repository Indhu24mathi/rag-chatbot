from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_answer(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    pages_used = list(set([
        str(doc.metadata.get("page", 0) + 1)
        for doc in relevant_docs
    ]))

    prompt = (
        "You are a helpful assistant. "
        "Answer the question based ONLY on the context below. "
        "If the answer is not in the context, say: "
        "I could not find that in the document.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, pages_used