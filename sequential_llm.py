from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


os.environ['LANGCHAIN_PROJECT'] = 'sequentila llm app'

load_dotenv()
model1 = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0.2)
model2 = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
