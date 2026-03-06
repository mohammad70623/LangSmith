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
from langsmith import traceable

os.environ['LANGCHAIN_PROJECT'] = 'Rag Chatbot'
load_dotenv()  

DATA_PATH = "data/"

@traceable(name="load_pdf")
def load_pdf_files(data): 
    loader = DirectoryLoader(data, 
                            glob = '*.pdf', 
                            loader_cls = PyPDFLoader)

    documents = loader.load() 
    return documents
documents = load_pdf_files(data=DATA_PATH)

@traceable(name="split_documents")
def creat_chunks(extracted_data): 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, 
                                                  chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks 
text_chunks = creat_chunks(extracted_data = documents)

@traceable(name="Create_embedding")
def get_embedding_model(): 
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model = get_embedding_model()


db = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding_model,
    persist_directory="vectorstore/db_chroma" 
)
retriever = db.as_retriever(search_kwargs={"k": 3})


prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)  
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()


print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)