from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0, 
    model="gpt-3.5-turbo-0613", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


restaurant_data = [
    {"name": "Namaste", "description": "A vegetarian-friendly Indian restaurant in Brussels.", "city": "Brussels", "tags": ["vegetarian"],"address": "Rue Jules Van Praet 30, Brussels 1000, Belgium"},
    {"name": "Spice of India", "description": "Non-vegetarian Indian restaurant in Brussels with spicy dishes.", "city": "Brussels", "tags": ["spicy"]},
    {"name": "Maharaja Palace", "description": "A luxurious Indian restaurant in Brussels.", "city": "Brussels", "tags": ["luxurious"]},
    {"name": "Feux Du Bengale", "description": "Famous for extensive menus with flavorful and delicious meals for foodies craving Indian classics cooked traditionally. Specialties include chicken and veg biryani.", "city": "Brussels", "tags": ["traditional"], "address": "Rue des Eperonniers 69, Brussels 1000, Belgium"},
    {"name": "La Porte Des Indes", "description": "Beautiful restaurant bringing back the glory of exquisite Indian cuisines. Offers divine chicken and biryanis, as well as alluring sweet dishes.", "city": "Brussels", "tags": ["chicken"], "address": "Avenue Louise 455, Brussels 1050, Belgium"},
    {"name": "Indian Mixed Grill", "description": "Affordable place offering chicken tikka, special lamb biryani, and eggplant side dishes. Finger-licking good food!", "city": "Ganshoren", "tags": ["chicken"], "address": "Avenue Charles-Quint 157, Ganshoren, Brussels 1083, Belgium"},
    {"name": "Shezan", "description": "Beautifully decorated restaurant with superb hospitality and tempting food dishes. Offers poppadoms with delicious chutneys and tikkas.", "city": "Brussels", "tags": ["chutneys"], "address": "Chaussee de Wavre near 120| Near porte de namur, Brussels 1050, Belgium"},
    {"name": "Mumtaz", "description": "Place conquered by Indian spices and flavors. Set in a foreign country, this place will be your greatest discovery.", "city": "Brussels", "tags": ["spicy"], "address": "Chaussee de Wavre 64, Brussels 1050, Belgium"},
    {"name": "Garden of Punjab", "description": "Restaurant serving authentic, appetizing and inviting experiences. Food served with dollops of love and special requests granted.", "city": "Ixelles", "tags": ["spicy"], "address": "Avenue Adolphe Buyl 19, Ixelles, Brussels 1050, Belgium"},
    {"name": "Le Taj", "description": "Tiny yet casual and warm place catering to personalized needs of Indian recipes and dishes. Known for the greatest chicken korma of all times.", "city": "Saint-Gilles", "tags": ["chicken"], "address": "Rue de l'hôtel des Monnaies 33, Saint-gilles, Brussels 1060, Belgium"}
]

restaurant_descriptions = [restaurant['description'] for restaurant in restaurant_data]

vectorstore = FAISS.from_texts(texts=restaurant_descriptions, embedding=embedding_model)

text_template = """
    Chat history: 
    {chat_history}
You are a restaurant retriever , you will give suggestions to users about Indian restaurants in Brussels, Belgium.
"""
template = ChatPromptTemplate.from_messages([
    ("system", text_template),
    ("human", "{input}")
])

chat_history = ""

while True:
    user_query = input("Please enter your query (or type 'exit' to quit): ")
    
    if user_query.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    formatted_query = template.format(input=user_query, chat_history=chat_history)
    response = llm.invoke(formatted_query)


    llm_response_content = response.content
    query_embedding = embedding_model.embed_query(user_query)
    search_results = vectorstore.similarity_search(query=user_query, k=3)
    user_keywords = user_query.lower().split()
   
    relevant_results = []
    for result in search_results:
        description = result.page_content
        if description in restaurant_descriptions:
            index = restaurant_descriptions.index(description)
            if any(tag in user_keywords for tag in restaurant_data[index]['tags']):
                relevant_results.append(restaurant_data[index])

    combined_response = f"{llm_response_content}\n\nTop restaurant suggestions based on your query:\n"
    if relevant_results:
        for restaurant in relevant_results:
            combined_response += f"Name: {restaurant['name']}, Description: {restaurant['description']}, City: {restaurant['city']}\n"
    else:
        combined_response += "No matching restaurants found."

    print("\n" + combined_response)

    chat_history += f"\nUser: {user_query}\nAssistant: {llm_response_content}\n"