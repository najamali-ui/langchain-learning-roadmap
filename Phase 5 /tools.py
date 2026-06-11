import os
import requests
import yfinance as yf
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import OllamaEmbeddings

@tool
def get_stock_price(ticker: str) -> str:
    """Fetch the real-time stock price and basic info for a given ticker symbol (e.g., AAPL, MSFT)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        if current_price is None:
            return f"Could not retrieve price for {ticker}. Please check the ticker symbol."
        
        name = info.get('shortName', ticker)
        return f"The current stock price of {name} ({ticker}) is ${current_price}."
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {e}"

@tool
def get_weather(location: str) -> str:
    """Fetch the current weather for a specific location."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if api_key and api_key != "your_openweather_api_key_here":
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                desc = data['weather'][0]['description']
                return f"The current weather in {location} is {temp}°C with {desc}."
            else:
                return f"Could not get weather for {location} from OpenWeatherMap."
        except Exception as e:
            return f"Error with OpenWeatherMap: {e}"
    else:
        # Fallback to DuckDuckGo
        search = DuckDuckGoSearchRun()
        query = f"current weather in {location}"
        try:
            result = search.invoke(query)
            return result
        except Exception as e:
            return f"Error fetching weather data: {e}"

def get_faq_retriever_tool():
    """Initializes and returns a retriever tool for the TechStore FAQ vector database."""
    embeddings = OllamaEmbeddings(model="qwen")
    
    retriever = None
    faiss_path = os.path.join(os.path.dirname(__file__) or ".", "faiss_index")
    
    if os.path.exists(faiss_path):
        try:
            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            print("Using FAISS for FAQ retrieval.")
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
    
    if retriever is None:
        print("WARNING: FAQ retriever could not be initialized. Run ingest.py first.")
        @tool
        def dummy_faq_tool(query: str) -> str:
            """Search the FAQ knowledge base."""
            return "The FAQ knowledge base is currently unavailable. Please run ingest.py first to build the index."
        return dummy_faq_tool

    from langchain_core.tools.retriever import create_retriever_tool
    faq_tool = create_retriever_tool(
        retriever,
        "techstore_faq_search",
        "Search for information about TechStore policies, products, returns, and other frequently asked questions. Always use this tool first when a user asks a general question about the store."
    )
    return faq_tool

