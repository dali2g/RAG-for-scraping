from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM with your API key (ensure you have set it in your .env file)
llm = ChatOpenAI(
    temperature=0, 
    model="gpt-3.5-turbo-0613", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Text template for the conversation
text_template = """
    Chat history: 
    {chat_history}
You are an assistant, you will give suggestions to users about Indian restaurants in Brussels, Belgium.
"""

# Create a ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", text_template),
    ("human", "{input}")
])

# Initialize chat history
chat_history = ""

# Start the conversation loop
while True:
    # Prompt the user to enter a query
    user_query = input("Please enter your query (or type 'exit' to quit): ")
    
    # Exit the loop if the user types 'exit'
    if user_query.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    # Format the query using the template and include both the user input and chat history
    formatted_query = template.format(input=user_query, chat_history=chat_history)

    # Get the response from the LLM
    response = llm.invoke(formatted_query)

    # Print both the query and the response
    print("User Query:", user_query)
    print("LLM Response:", response)

    # Update the chat history
    chat_history += f"\nUser: {user_query}\nAssistant: {response}\n"
