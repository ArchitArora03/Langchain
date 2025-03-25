from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation")
model=ChatHuggingFace(llm=llm)

parser= StrOutputParser()


prompt1 = PromptTemplate(
    template='Generate a detailed report on {country} economy',
    input_variables=['country']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result= chain.invoke ({'country': 'USA'})

print(result)

chain.get_graph().print_ascii()