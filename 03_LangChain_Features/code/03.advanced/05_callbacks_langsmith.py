# Use Case: Callbacks and LangSmith
# - Monitoring and debugging travel recommendation systems
# - Logging interactions with travel planning assistants
# - Evaluating the performance of travel information retrieval systems

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.tracers import LangChainTracer # For LangSmith
from langchain.callbacks.handlers import StdOutCallbackHandler # For StdOut

# Load environment variables at the module level
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please ensure you have the necessary API keys in your .env file")
    print("You can install dependencies with: python3 -m pip install -r requirements.txt")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_callbacks_and_langsmith_info():
    print("\n--- Travel Recommendation System with Monitoring Demo ---")
    print("A demonstration of how to monitor and debug travel recommendation systems")
    
    # Step 1: Set up the embeddings model and vector store
    print("\nInitializing the travel knowledge base...")
    embeddings = OpenAIEmbeddings()
    
    # Create a travel knowledge base for our demonstration
    travel_knowledge = """
## Destination: Kyoto, Japan
Kyoto is the cultural heart of Japan, featuring over 1,600 Buddhist temples and 400 Shinto shrines.
Best time to visit: Spring (March-April) for cherry blossoms or Fall (October-November) for autumn colors.
Popular attractions include Kinkaku-ji (Golden Pavilion), Fushimi Inari Shrine, and the historic Gion district.
Kyoto is known for traditional kaiseki cuisine, matcha tea, and yudofu (tofu hot pot).
Accommodation options range from luxury hotels to traditional ryokans (Japanese inns).
The city has an efficient bus system, and many attractions are accessible by bicycle.

## Destination: Barcelona, Spain
Barcelona is the cosmopolitan capital of Spain's Catalonia region, known for art and architecture.
Best time to visit: May to June or September to October for pleasant weather and fewer crowds.
Must-see attractions include Sagrada Familia, Park Güell, La Rambla, and the Gothic Quarter.
The city is famous for tapas, paella, and Catalan cuisine.
Barcelona has excellent public transportation with an extensive metro system.
Beach lovers can enjoy Barceloneta Beach, just minutes from the city center.

## Destination: Cape Town, South Africa
Cape Town sits at the southern tip of Africa, known for its stunning natural beauty.
Best time to visit: March to May or September to November for mild weather.
Table Mountain, Cape of Good Hope, and Robben Island are major attractions.
The city is a food lover's paradise, with fresh seafood and diverse culinary influences.
The Cape Winelands, just outside the city, offer world-class wine tasting experiences.
Renting a car is recommended for exploring the surrounding areas.
"""
    
    # Split the travel knowledge into chunks for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    knowledge_chunks = text_splitter.create_documents([travel_knowledge])
    
    # Create the vector store and retriever
    vector_store = FAISS.from_documents(knowledge_chunks, embeddings)
    travel_retriever = vector_store.as_retriever()
    print(f"Travel knowledge base created with {len(knowledge_chunks)} chunks")
    
    # Step 2: Set up the callbacks for monitoring
    print("\nSetting up monitoring system...")
    
    # Create a standard output callback handler to show the chain's operations
    stdout_handler = StdOutCallbackHandler()
    
    print("\nNote on LangSmith integration:")
    print("For production travel recommendation systems, you would likely use LangSmith")
    print("for comprehensive monitoring, evaluation, and debugging capabilities.")
    print("This requires setting up a LangSmith account and API keys.")
    
    # Step 3: Create a travel recommendation QA system with callbacks
    print("\nInitializing travel recommendation system with monitoring...")
    travel_qa_with_logging = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=travel_retriever,
        callbacks=[stdout_handler]  # This will print detailed logs to the console
    )
    
    print("Travel recommendation system initialized with monitoring!")
    print("----------------------------------------\n")
    
    # Step 4: Demonstrate the monitored travel recommendation system
    print("--- Travel Recommendation System Demonstration ---")
    print("\nObserve the detailed logs as the system processes the query:\n")
    
    # Ask a travel question and observe the callbacks in action
    travel_query = "What's the best time to visit Kyoto and what attractions should I see?"
    print(f"Travel Query: {travel_query}")
    
    # The StdOutCallbackHandler will print details about each step in the chain
    result = travel_qa_with_logging.invoke({"query": travel_query})
    
    # Display the final result after all the callback logs
    print(f"\nFinal Travel Recommendation: {result['result']}")
    
    # Step 5: Provide information about LangSmith for advanced monitoring
    print("\n--- LangSmith for Advanced Travel System Monitoring ---")
    print("For production travel planning systems, LangSmith provides comprehensive")
    print("monitoring, evaluation, and debugging capabilities.")
    print("\nTo enable LangSmith for your travel recommendation system:")
    print("1. Sign up at https://smith.langchain.com/")
    print("2. Create an API Key from the LangSmith settings page")
    print("3. Set these environment variables in your .env file:")
    print("   LANGCHAIN_TRACING_V2=true")
    print("   LANGCHAIN_API_KEY='YOUR_LANGSMITH_API_KEY'")
    print("   LANGCHAIN_PROJECT='Travel_Recommendation_System'")
    print("\nOnce configured, you can:")
    print("- Monitor your travel system's performance in real-time")
    print("- Debug complex travel planning chains")
    print("- Evaluate recommendation quality with feedback loops")
    print("- Compare different travel recommendation models")
    print("----------------------------------------\n")
    
    # Step 6: Interactive demo with simple console callback
    print("\n--- Interactive Travel Assistant with Logging ---")
    print("Ask travel questions and see both the internal operations and final response.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_input = input("\nYour travel question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("Please enter a question.")
            continue
        
        try:
            # Process the user's travel question with the monitored system
            print("\nProcessing your travel question (with logging):\n")
            result = travel_qa_with_logging.invoke({"query": user_input})
            print(f"\nTravel Assistant: {result['result']}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for trying the Travel Recommendation System with Monitoring!")

if __name__ == "__main__":
    run_callbacks_and_langsmith_info()
    print("Callbacks and LangSmith example finished.")
