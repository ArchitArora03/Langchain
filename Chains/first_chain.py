from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")
model=ChatHuggingFace(llm=llm)

prompt = PromptTemplate( 
    template='Best cities to visit in {country}',
    input_variables=['country']
)
parser=StrOutputParser()

chain = prompt | model | parser 

result=chain.invoke({'country': 'France'})

print(result)
