from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm1=HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta', task="text-generation")
llm2=HuggingFaceEndpoint(repo_id='HuggingFaceH4/zephyr-7b-beta',task="text-generation")
model1=ChatHuggingFace(llm=llm1)
model2=ChatHuggingFace(llm=llm2)

prompt1=PromptTemplate(
    template='Create short notes using the {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Create a 5 questions quiz using the {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the notes and the 5 quiz questions into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1| model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain= prompt3 | model1 | parser
text= '''
LangChain is a powerful framework designed for developing applications that leverage large language models (LLMs) like OpenAI’s GPT, through a modular approach. It allows developers to build end-to-end applications that integrate LLMs with various tools, APIs, databases, and more, enabling them to create complex workflows with ease. LangChain provides various abstractions like *agents*, *chains*, *prompts*, and *memory*, which help organize and structure the flow of data, interactions, and decisions within a program. Chains enable the chaining of multiple calls to models or tools in sequence, while agents help with decision-making processes, selecting tools dynamically based on input. It also integrates with external services and platforms, such as cloud storage or web scraping tools, allowing users to extend the capabilities of their applications. LangChain makes it easy for developers to handle data flows, interaction management, and even long-running tasks, all while simplifying the process of working with LLMs by offering ready-to-use components and patterns for common problems. This flexibility makes LangChain a popular choice for building more sophisticated AI-driven applications, such as chatbots, personal assistants, or automated content generation systems.
'''

chain= parallel_chain | merge_chain


result= chain. invoke ({'text': text})

print(result)

chain.get_graph().print_ascii()