import os
import openai
import settings
import psycopg2
from langchain.document_loaders import UnstructuredMarkdownLoader,UnstructuredFileLoader,TextLoader,PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, AzureOpenAI
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from pdf2embed_langchain import (clean_text,
                                 preprocess_document,
                                 pdf_loader,
                                 chunking,
                                 embedding_store
                                 )
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores import PGEmbedding
from text_extractor.extractor import TextExtractor
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from pathlib import Path
from langchain.callbacks import get_openai_callback

def chatbot(llm,vector_store):
    conversational_mem = ConversationBufferMemory(memory_key="chat_history",k=10,return_messages=True)#, input_key='input', output_key="output")
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(),
    )

    tools = [
        Tool(
            name="UKP",
            func = qa.run,
            description=(
                'use this tool for any document ingestion to having a QA session'
            )
        )
    ]

    agent = initialize_agent(
        agent = 'chat-conversational-react-description',
        tools = tools,
        llm=llm,
        verbose=False,
        max_iterations = 3,
        early_stopping_method = 'generate',
        memory=conversational_mem,
        # return_source_documents = True,
        # return_intermediate_steps=True
    )
    while True:
        query = input("What do you want to discuss about?\n\n")
        if query.lower() == "exit":
            break
        with get_openai_callback() as cb:
            response = agent(query)
            print(response["output"])
            print(f"\n{cb.total_tokens}")


def talk2doc(llm,vector_store):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(),
        return_source_documents = True
    )
    
    
    while(True):
        query = input("How may i help you?\n\n")
        if query.lower() == "exit":
            break
        with get_openai_callback() as cb:
            response = qa({"query":query})
            print(f"AI Response: {response['result']}")
            print(f"\n{cb.total_tokens}")
        # print(f"source: {response['source_documents'][0]}")
        # print("\n")

def clear_db():  
    conn = psycopg2.connect(  
        database=settings.name,  
        user=settings.user,  
        password=settings.password,  
        host=settings.host,  
        port=settings.port  
    )   
    cur = conn.cursor()    
    cur.execute("DELETE FROM langchain_pg_embedding")   
    conn.commit()    
    cur.close()  
    conn.close()  


def main():
    print("\n\nHello! Welcome to SimpleBot.\nHow may I help you today?")
    print("\nPlease provide the path to your document")
    clear_db()
    file_path = input()
    print("\nThank you, processing your document...")
    info_data = extract(file_path)
    openai_api_key = settings.openai_api_key

    doc_pdf = Document(page_content = info_data[str(os.path.basename(Path(file_path)))]["data"], metadata=info_data[str(os.path.basename(file_path))]["metadata"])
    chunked_docs = chunking(doc_pdf)
    _,model = embedding_store("quivr",chunked_docs)
    llm = AzureChatOpenAI(deployment_name = "GPT35Turbo",openai_api_key= openai_api_key,model_name="GPT35Turbo", openai_api_base="https://msoaopenai.intel.com",openai_api_version="2023-05-15", openai_api_type="azure")

    host = settings.host
    port = settings.port
    user = settings.user
    name = settings.name
    password = settings.password

    vector_store = PGVector(
        embedding_function=model,
        collection_name=name,
        connection_string=f"postgresql://{user}:{password}@{host}:{port}/{name}"
    )
    
    print("Ok! It's a interesting document.\n\nWould you like to 1. ask me question on the document or 2. you want to discuss with me on this document?")
    # option = input("Choose 1 or 2\n")
    if True:
        talk2doc(llm,vector_store)
    elif int(option) == 2:
        chatbot(llm,vector_store)
    else:
        print("invalid option")

def extract(folder_path):
    extractor = TextExtractor(folder_path, "output.json")
    if os.path.isfile(folder_path):
        extracted_data = extractor.extract_text_from_single_document()
    else:
        extracted_data = extractor.extract_text_from_documents()
    # extractor.save_to_json(extracted_data)
    return extracted_data

if __name__ == "__main__":
    main()
