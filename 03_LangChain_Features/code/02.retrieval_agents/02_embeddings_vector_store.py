# Use Case: Embeddings and Vector Store
# - Converting text into semantic vector representations
# - Creating efficient search databases for natural language queries
# - Finding similar content without relying on exact keyword matching

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_embeddings_vector_store():
    print("\n--- Embeddings & Vector Store Demo ---")
    
    # Step 1: Initialize the embeddings model
    print("\nInitializing OpenAI Embeddings model...")
    embeddings = OpenAIEmbeddings()
    print("Embeddings model initialized successfully!")
    
    # Step 2: Load and prepare travel destination data
    print("\nLoading travel destination data...")
    destinations_data = """
        # POPULAR TRAVEL DESTINATIONS

        ## Paris, France
        Paris, the City of Light, is known for iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city offers world-class cuisine, fashion, and art scenes. Visitors enjoy strolling along the Seine River, exploring charming neighborhoods like Montmartre, and experiencing Parisian café culture. Best visited in spring (April-June) or fall (September-October) to avoid summer crowds.

        ## Barcelona, Spain
        Barcelona combines stunning architecture, 600 beautiful churches, and vibrant culture. Antoni Gaudí's masterpieces, including Sagrada Familia and Park Güell, showcase the city's unique aesthetic. La Rambla street and Gothic Quarter offer excellent shopping and dining, while Barceloneta Beach provides Mediterranean relaxation. The city is famous for tapas, seafood paella, and Catalan cuisine.

        ## Tokyo, Japan
        Tokyo is a fascinating mix of ultramodern and traditional Japanese culture. Visit the bustling Shibuya Crossing, serene Meiji Shrine, and the Imperial Palace. The city offers incredible shopping in Ginza and Harajuku, world-renowned sushi and ramen, and beautiful cherry blossoms in spring. Tokyo's efficient public transportation makes it easy to explore different neighborhoods.

        ## Bali, Indonesia
        Bali is a tropical paradise known for lush rice terraces, volcanic mountains, and pristine beaches. The island combines natural beauty with rich cultural experiences through its Hindu temples, traditional dance performances, and craft villages. Popular areas include Ubud for culture, Seminyak for dining and nightlife, and Nusa Dua for luxury resorts. The dry season (April-October) is the best time to visit.

        ## New York City, USA
        New York City offers iconic attractions like Times Square, Central Park, and the Statue of Liberty. The city's diverse neighborhoods include Manhattan's skyscrapers, Brooklyn's trendy scenes, and Queens' international food options. World-class museums (MoMA, Metropolitan), Broadway shows, and endless shopping and dining make it a premier urban destination. Visit during spring or fall for comfortable weather.
    """
    
    # Step 3: Split data into chunks
    print("Splitting destination data into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        separators=["\n## ", "\n\n", "\n", ".", "!", "?"],
        length_function=len
    )
    destination_chunks = text_splitter.create_documents([destinations_data])
    print(f"Created {len(destination_chunks)} document chunks")
    
    # Step 4: Create vector embeddings and store in FAISS
    print("\nCreating vector embeddings and building search index...")
    vector_store = FAISS.from_documents(destination_chunks, embeddings)
    print("Vector store created successfully!")
    
    # Step 5: Demonstrate similarity search with travel queries
    print("\n--- Semantic Search Demonstration ---")
    
    # Example 1: Search for beach destinations
    print("\nQuery 1: 'Which destinations have nice beaches?'")
    beach_results = vector_store.similarity_search(
        "Which destinations have nice beaches?", k=2
    )
    print("Results:")
    for i, doc in enumerate(beach_results):
        print(f"\nResult {i+1}:")
        print(doc.page_content)
        print("-" * 40)
    
    # Example 2: Search for cultural experiences
    print("\nQuery 2: 'I'm interested in art and museums'")
    culture_results = vector_store.similarity_search(
        "I'm interested in art and museums", k=2
    )
    print("Results:")
    for i, doc in enumerate(culture_results):
        print(f"\nResult {i+1}:")
        print(doc.page_content)
        print("-" * 40)
    
    return vector_store

def interactive_travel_search(vector_store):
    print("\n--- Interactive Travel Search ---")
    print("Ask questions about travel destinations. Type 'exit' or 'quit' to end.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("Please enter a question.")
            continue
        
        # Step 1: Search for relevant chunks based on query
        search_results = vector_store.similarity_search(user_input, k=2)
        
        if not search_results:
            print("\nAI: I don't have information about that in my travel database.")
            continue
        
        # Step 2: Create context from search results
        context = "\n\n".join([doc.page_content for doc in search_results])
        
        # Step 3: Generate response using LLM with retrieved context
        print(f"\nContext: {context}")
        print(f"\nUser Input: {user_input}")
        prompt = f"Based on this travel information:\n\n{context}\n\nAnswer this question: {user_input}\n\nOnly use information from the provided context. If the answer isn't in the context, say you don't have that specific information."
        
        try:
            response = llm.invoke(prompt)
            print(f"\nAI: {response.content}")
        except Exception as e:
            print(f"\nAI: Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    vector_store = run_embeddings_vector_store()
    interactive_travel_search(vector_store)
    print("\nEmbeddings & Vector Store demo completed. Thank you for using the Travel Planning Assistant!")

# Travel Planning Assistant - Practical Example

"""
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 1. Load hotel data from CSV file
print("Loading hotel database from CSV...")
hotel_loader = CSVLoader(file_path='hotels_database.csv')
hotel_data = hotel_loader.load()
print(f"Loaded information for {len(hotel_data)} hotels")

# 2. Create embeddings and vector store with metadata
embeddings = OpenAIEmbeddings()
vector_db = Chroma.from_documents(
    documents=hotel_data,
    embedding=embeddings,
    collection_name="hotels_database",
    metadata_config={"hotel_name": "string", "location": "string", "price_range": "string", "amenities": "string"}
)

...
...
...

"""

# Summary: This file demonstrates how to create and use embeddings with vector stores for semantic
# search. It shows how to convert text chunks into vector embeddings, store them in a FAISS index,
# and perform similarity searches to find relevant information based on natural language queries.
# The example uses travel destination data to showcase how semantic search can power a travel
# recommendation system.
