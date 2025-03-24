from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template=ChatPromptTemplate([
    ('system','You are an helful customer support agent'),
    MessagesPlaceholder(variable_name='Chat_History'),
    ('human', '{query}')
])

chat_history=[]
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt=chat_template.invoke({'Chat_History': chat_history, 'query':'Where is my order'})

print(prompt)