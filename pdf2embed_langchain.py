import os
import openai
import settings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from langchain.vectorstores.pgvector import PGVector

openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://msoaopenai.intel.com"
openai.api_version = "2023-05-15"

host = settings.host
port = settings.port
user = settings.user
name = settings.name
password = settings.password
openai_api_key = settings.openai_api_key


def clean_text(text: str) -> str:
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translation_table)

    words = word_tokenize(text)

    filler_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in filler_words]

    text = ' '.join(words)
    return text

def preprocess_document(chunks):
    for doc in chunks:
        doc.page_content = clean_text(doc.page_content.replace('\x00', ''))
    return chunks

def pdf_loader(file):
    loader_pdf = PyPDFLoader(file)
    doc_pdf = loader_pdf.load()
    return doc_pdf

def chunking(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000,chunk_overlap=200)
    chunk_docs = text_splitter.split_documents([docs])
    return preprocess_document(chunk_docs)

def embedding_store(collection_name,chunks):
    embedding_model = OpenAIEmbeddings(deployment="text-embedding-ada-002",openai_api_key= openai_api_key, openai_api_base="https://msoaopenai.intel.com",openai_api_version="2023-05-15",openai_api_type="azure")
    i = 0
    total_chunks = len(chunks)
    while i < total_chunks:
        if i+16 < total_chunks:
            db = PGVector.from_documents(
                embedding=embedding_model,
                documents=chunks[i:i+16],
                collection_name=collection_name,
                connection_string=f"postgresql://{user}:{password}@{host}:{port}/{name}"
            )
            i+=16
        else:
            db = PGVector.from_documents(
                embedding=embedding_model,
                documents=chunks[i:total_chunks],
                collection_name=collection_name,
                connection_string=f"postgresql://{user}:{password}@{host}:{port}/{name}"
            )
            i = total_chunks
    return db,embedding_model

# if __name__ == "__main__":
#     doc_pdf = pdf_loader('./langchain_experiments/FILES/test.pdf')
#     chunked_docs = chunking(doc_pdf)
#     conn = embedding_store("quivr",chunked_docs)
#     print("Done")


