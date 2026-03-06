import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'Rag Chatbot'
load_dotenv()  

DATA_PATH = "data/"
def load_pdf_files(data): 
    loader = DirectoryLoader(data, 
                            glob = '*.pdf', 
                            loader_cls = PyPDFLoader)

    documents = loader.load() 
    return documents
documents = load_pdf_files(data=DATA_PATH)

