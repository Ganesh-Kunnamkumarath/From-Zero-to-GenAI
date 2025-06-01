# Use Case: Retrieval QA Chain
# - Providing direct answers to user questions by retrieving relevant information from a knowledge base
# - Creating travel information systems that can answer specific questions about destinations
# - Allowing users to get accurate information without having to search through large documents

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def run_retrieval_qa():
    # Load environment variables
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None:
        print("Error: OPENAI_API_KEY environment variable not set.")
        exit()

    print("\n--- Travel Information Retrieval System Demo ---")
    
    # Step 1: Initialize the embeddings model
    print("\nInitializing embeddings model and vector store...")
    # Initialize LLM and Embeddings
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    embeddings = OpenAIEmbeddings()
    
    # Step 2: Create a knowledge base with travel information
    travel_guide = """
## Paris, France
Paris, known as the City of Light, is famous for its iconic landmarks including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. The best times to visit are spring (April-June) or fall (September-October) when the weather is mild and crowds are smaller. The Paris Metro provides excellent public transportation throughout the city. The currency used is the Euro (€). While French is the official language, many people in tourist areas speak English. Popular activities include taking a Seine River cruise, visiting Montmartre for panoramic city views, and enjoying French cuisine at local bistros. Visitors should be aware of pickpockets in tourist areas and note that many shops close on Sundays.

## Tokyo, Japan
Tokyo is Japan's bustling capital, blending ultramodern and traditional elements. Key attractions include the Shibuya Crossing, Tokyo Skytree, and the historic Senso-ji Temple in Asakusa. Cherry blossom season (late March to early April) is especially beautiful but crowded. Fall (October-November) offers pleasant weather and colorful foliage. The currency is the Japanese Yen (¥). Japanese is the main language, with limited English in tourist areas. The train and subway system is efficient but can be complex for first-time visitors. Tokyo is known for its exceptional food scene, from high-end sushi restaurants to casual ramen shops. Tipping is not customary in Japan, and it's considered polite to bow when greeting people.

## New York City, USA
New York City features iconic sites like Times Square, Central Park, and the Statue of Liberty. The city has a diverse food scene with cuisines from around the world. Spring (April-June) and fall (September-November) offer the most pleasant weather. The US Dollar ($) is the currency. English is the main language. The extensive subway system runs 24/7 and is the easiest way to navigate the city. NYC is divided into five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island, each with its own character. Broadway shows are a major attraction, with tickets available at various price points. New York experiences four distinct seasons, with hot summers and cold, sometimes snowy winters.

## Bali, Indonesia
Bali is an Indonesian island known for its forested volcanic mountains, iconic rice paddies, beaches, and coral reefs. The dry season (April to October) is the best time to visit. The Indonesian Rupiah (IDR) is the currency. Balinese and Indonesian are spoken, with English common in tourist areas. Popular destinations include Ubud for culture and arts, Kuta and Seminyak for beaches and nightlife, and the Uluwatu Temple for spectacular ocean views. Balinese cuisine features aromatic spices, fresh vegetables, and seafood. Visitors should dress modestly when visiting temples and always remove shoes before entering sacred places. Renting a scooter is a popular way to get around, though traffic can be chaotic.

## Sydney, Australia
Sydney is known for its stunning harbor, the Sydney Opera House, and beautiful beaches like Bondi. Summer (December-February) is peak season, though spring (September-November) offers pleasant weather with fewer crowds. The Australian Dollar (AUD) is the currency. English is the main language. Public transport includes trains, buses, and ferries. The Harbour Bridge Climb provides spectacular views of the city. Sydney's diverse neighborhoods range from the historic Rocks district to the trendy Surry Hills. Australian cuisine in Sydney features fresh seafood, quality meats, and multicultural influences. Visitors should apply sunscreen regularly, even on cloudy days, as Australia has high UV levels. Most beaches have lifeguards, and swimmers should always stay between the red and yellow flags.
"""
    
    # Create document chunks and vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    travel_guide_chunks = text_splitter.create_documents([travel_guide])
    vector_store = FAISS.from_documents(travel_guide_chunks, embeddings)
    doc_retriever = vector_store.as_retriever()
    print(f"Travel guide knowledge base created with {len(travel_guide_chunks)} chunks")
    
    # Step 3: Create a Retrieval QA chain
    print("\nSetting up the travel information retrieval system...")
    travel_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff documents into a single prompt
        retriever=doc_retriever,
        return_source_documents=True  # Return source information for transparency
    )
    print("Travel information retrieval system initialized!")
    print("----------------------------------------\n")
    
    # Step 4: Demonstrate with example queries
    print("--- Travel Information System Demonstration ---")
    
    # Example 1: Best time to visit query
    query1 = "When is the best time to visit Tokyo?"
    print(f"\nQuery: {query1}")
    result1 = travel_qa_chain.invoke({"query": query1})
    print(f"Answer: {result1['result']}")
    if result1.get('source_documents'):
        print(f"Source: Information from the {result1['source_documents'][0].page_content[:50]}...")
    
    # Example 2: Currency query
    query2 = "What currency is used in Bali?"
    print(f"\nQuery: {query2}")
    result2 = travel_qa_chain.invoke({"query": query2})
    print(f"Answer: {result2['result']}")
    if result2.get('source_documents'):
        print(f"Source: Information from the {result2['source_documents'][0].page_content[:50]}...")
    print("----------------------------------------\n")
    
    print("Retrieval QA with Travel Knowledge Base example finished.")
    
    print("\n--- Interactive Travel Information Assistant ---")
    print("Ask any questions about Paris, Tokyo, New York, Bali, or Sydney.")
    print("Type 'exit' or 'quit' to end the session.")
    
    try:
        # Use the same LLM as before for consistency
        chat_llm = travel_qa_chain
        print("Interactive Travel Assistant Initialized.")
    except Exception as e:
        print(f"Error initializing Travel Assistant: {e}")
        print("Please ensure your OPENAI_API_KEY is correctly set in the .env file.")
        chat_llm = None

    if chat_llm: 
        while True:
            user_query = input("Your question: ")
            if user_query.lower() in ["exit", "quit"]:
                break
            if not user_query.strip():
                print("Please enter a question.")
                continue
            
            try:
                result = chat_llm.invoke({"query": user_query})
                print(f"\nAnswer: {result['result']}")
                if result.get('source_documents'):
                    print(f"Source: Based on information about {result['source_documents'][0].page_content.split('##')[1].strip() if '##' in result['source_documents'][0].page_content else 'various destinations'}")
            except Exception as e:
                print(f"Error: {e}")

    print("Exiting interactive travel assistant.")
    print("\nThank you for using the Travel Information Retrieval System!")

if __name__ == "__main__":
    run_retrieval_qa()
