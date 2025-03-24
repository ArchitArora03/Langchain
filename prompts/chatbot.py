from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm= HuggingFaceEndpoint(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',task='text-generation')
model=ChatHuggingFace(llm=llm)

while True:
    user_input=input('You: ')
    if user_input=='Exit':
        break
    else:
        result=model.invoke(user_input)
        print('AI:',result.content)
    