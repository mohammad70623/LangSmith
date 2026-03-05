from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv() 
model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0.2)

prompt = PromptTemplate.from_template("{question}")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of Bangladesh"})
print(result)