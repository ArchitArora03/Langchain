from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm= HuggingFaceEndpoint(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',task='text-generation')
model=ChatHuggingFace(llm=llm)

chat_history=[]
while True:
    user_input=input('You: ')
    chat_history.append(user_input)
    if user_input=='Exit':
        break
    else:
        result=model.invoke(chat_history)
        chat_history.append(result.content)
        print('AI:',result.content)

print(chat_history)
    