from langchain_core.runnables.base import RunnableSequence
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def scrape_restaurants(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract restaurant details
    restaurants = []
    for restaurant in soup.select('.listing_title'):
        name = restaurant.text.strip()
        link = "https://www.tripadvisor.com" + restaurant.a['href']
        restaurants.append(f"{name}: {link}")
    
    # Return the data as a string
    return "\n".join(restaurants)

# Create the scraping tool
scraping_tool = Tool(
    name="RestaurantScraper",
    func=scrape_restaurants,
    description="Tool for scraping Indian restaurant information from Brussels TripAdvisor page."
)

# Initialize the LLM with your API key (ensure you have set it in your .env file)
llm = ChatOpenAI(
    temperature=0, 
    model="gpt-3.5-turbo-0613", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create the RunnableSequence
chain = RunnableSequence(
    scraping_tool, llm
)

# Example usage
url = "https://www.tripadvisor.com/Restaurants-g188644-c24-oa30-Brussels.html"
scraped_data = chain.invoke(url)
print(scraped_data)
