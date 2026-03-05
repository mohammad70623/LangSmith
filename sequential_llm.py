from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


os.environ['LANGCHAIN_PROJECT'] = 'sequentila llm app'

load_dotenv()
model1 = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0.2)
model2 = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

prompt1 = PromptTemplate(
   template = (
    "Create a comprehensive report on {topic} with the following structure:\n"
    "1. Introduction\n"
    "2. Background or context\n"
    "3. Key points or findings\n"
    "4. Examples or supporting data\n"
    "5. Conclusion"
),
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "Read the text below and generate a concise 5-point summary. Each point should be brief and informative.\n\n{text}",
    input_variables=['text']
)
