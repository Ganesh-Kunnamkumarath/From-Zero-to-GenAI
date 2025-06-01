# Use Case: Agents and Tools
# - Creating AI systems that can use multiple tools to solve complex tasks
# - Building assistants that can search for information and retrieve specific knowledge
# - Enabling language models to perform actions and make decisions autonomously

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchResults
# from langchain.tools import SerpAPIWrapper

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

# Global variable for vector_store
vector_store = None

def get_travel_info(destination: str) -> str:
    """Provides specific information about travel destinations from our knowledge base."""
    global vector_store # Access the global vector_store
    if not vector_store:
        return "Travel database not initialized. Cannot perform search."

    # Handle some popular destinations directly for faster response
    if "paris" in destination.lower():
        return "Paris, the City of Light, is known for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Best visited in spring (April-June) or fall (September-October) to avoid crowds. Famous for world-class cuisine, fashion, and art."
    
    elif "tokyo" in destination.lower():
        return "Tokyo is Japan's capital, blending ultramodern and traditional elements. Visit Shibuya Crossing, Meiji Shrine, and Tokyo Skytree. Known for excellent sushi, ramen, and shopping in districts like Ginza and Harajuku. Cherry blossom season (late March-early April) is especially beautiful."
    
    elif "bali" in destination.lower():
        return "Bali is an Indonesian island known for beautiful beaches, volcanic mountains, and spiritual retreats. Popular for surfing, yoga, and cultural experiences. Visit temples like Uluwatu and Tanah Lot, or explore rice terraces in Ubud. The dry season (April-October) is the best time to visit."
    
    else:
        # Fallback to vector store search if destination isn't in our preset responses
        results = vector_store.similarity_search(f"Information about traveling to {destination}", k=1)
        if results:
            return results[0].page_content
        return f"I don't have specific information about {destination} in my travel database, but an internet search might help."

def run_agents_and_tools():
    print("\n--- Travel Agent With Tools Demo ---")
    global vector_store
    
    # Step 1: Initialize embeddings model for our travel knowledge base
    print("\nInitializing tools and knowledge base...")
    embeddings = OpenAIEmbeddings()
    
    # Step 2: Load and prepare travel destination data for vector store
    travel_knowledge = """
        ## Paris, France
        Paris, the City of Light, is known for iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city offers world-class cuisine, fashion, and art scenes. Best visited in spring (April-June) or fall (September-October) to avoid summer crowds. The Paris Metro provides excellent public transportation. The currency is the Euro (€). French is the official language, though many in tourist areas speak English.

        ## Tokyo, Japan
        Tokyo is Japan's bustling capital, mixing ultramodern and traditional elements. Visit the iconic Shibuya Crossing and Tokyo Skytree for city views. The city is famous for sushi, ramen, and yakitori. The best times to visit are spring (March-May) for cherry blossoms or fall (September-November) for pleasant weather. The currency is the Japanese Yen (¥). Japanese is the primary language, with limited English in tourist areas. The efficient train and subway system makes transportation easy.

        ## Bali, Indonesia
        Bali is a tropical paradise known for beaches, volcanoes, and spiritual retreats. Popular activities include surfing, yoga, and visiting ancient temples. The dry season (April-October) is the best time to visit. The Indonesian Rupiah (IDR) is the currency. Balinese and Indonesian are spoken, with English common in tourist areas. Renting a scooter is a popular way to get around, though traffic can be chaotic.

        ## New York City, USA
        New York City features iconic sites like Times Square, Central Park, and the Statue of Liberty. The city offers world-class museums, Broadway shows, and diverse dining options. Spring (April-June) and fall (September-November) offer the most pleasant weather. The US Dollar ($) is the currency. English is the main language. The extensive subway system runs 24/7 and is the easiest way to navigate the city.

        ## Sydney, Australia
        Sydney is known for its stunning harbor, Opera House, and beautiful beaches like Bondi. The city offers excellent seafood, multicultural cuisine, and outdoor activities. Summer (December-February) is peak season, but spring (September-November) offers pleasant weather with fewer crowds. The Australian Dollar (AUD) is the currency. English is the main language. Public transport includes trains, buses, and ferries.
    """
    
    # Create document chunks and vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    knowledge_chunks = text_splitter.create_documents([travel_knowledge])
    vector_store = FAISS.from_documents(knowledge_chunks, embeddings)
    print(f"Travel knowledge base created with {len(knowledge_chunks)} destination chunks")
    
    # Step 3: Create tools for our travel agent to use
    print("\nSetting up travel planning tools...")
    
    # Tool 1: DuckDuckGo Search for up-to-date travel information
    search_tool = DuckDuckGoSearchResults()
    internet_search_tool = Tool(
        name="Internet Search",
        func=search_tool.run,
        description="Useful for finding current travel information such as flight prices, COVID restrictions, events, or recent travel advisories that might not be in our knowledge base."
    )

    # Tool 2: Custom Knowledge Base Tool for destination information
    destination_info_tool = Tool(
        name="Destination Information",
        func=get_travel_info,
        description="Useful for getting specific information about travel destinations like Paris, Tokyo, Bali, New York, and Sydney. Provides details about attractions, best times to visit, and local tips."
    )
    
    # Step 4: Initialize the Agent with our tools
    print("\nInitializing Travel Planning Agent...")
    tools = [internet_search_tool, destination_info_tool]
    
    travel_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,  # Show the agent's thought process
        handle_parsing_errors=True
    )
    print("Travel Planning Agent initialized successfully!")
    
    # Step 5: Demonstrate the agent with different travel queries
    print("\n--- Travel Agent Demonstration ---")
    
    # Example 1: Query that can be answered from our knowledge base
    print("\nQuery 1: 'What are the best things to do in Tokyo?'")
    response1 = travel_agent.invoke({"input": "What are the best things to do in Tokyo?"})
    print(f"\nTravel Agent Response:\n{response1['output']}")
    
    # Example 2: Query that likely needs internet search
    print("\nQuery 2: 'What's the current exchange rate between USD and Japanese Yen?'")
    response2 = travel_agent.invoke({"input": "What's the current exchange rate between USD and Japanese Yen?"})
    print(f"\nTravel Agent Response:\n{response2['output']}")
    
    # Example 3: Query that combines knowledge base and possibly search
    print("\nQuery 3: 'When is the best time to visit Bali and what should I pack?'")
    response3 = travel_agent.invoke({"input": "When is the best time to visit Bali and what should I pack?"})
    print(f"\nTravel Agent Response:\n{response3['output']}")
    
    return travel_agent

if __name__ == "__main__":
    run_agents_and_tools()
    print("Agents and Tools example finished.")

    print("\n--- Interactive Chat Mode ---")
    print("Type 'exit' or 'quit' to end the chat.")
    
    try:
        chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        print("Interactive LLM Initialized.")
    except Exception as e:
        print(f"Error initializing LLM for interactive mode: {e}")
        print("Please ensure your OPENAI_API_KEY is correctly set in the .env file.")
        chat_llm = None 

    if chat_llm: 
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip(): 
                print("AI: Please enter some text.")
                continue
            
            try:
                response = chat_llm.invoke(user_input)
                print(f"AI: {response.content}\n")
            except Exception as e:
                print(f"AI: Error during API call: {e}\n")

    print("Exiting interactive chat.")
