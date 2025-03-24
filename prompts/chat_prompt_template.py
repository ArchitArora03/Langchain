from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([
    ('system','You are a higly skilled {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt=chat_template.invoke({'domain':'Data Science', 'topic':'Langchain'})

print(prompt)