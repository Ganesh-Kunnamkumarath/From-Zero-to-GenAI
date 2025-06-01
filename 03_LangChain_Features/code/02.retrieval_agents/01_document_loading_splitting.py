# Use Case: Document Loading and Splitting
# - Processing large text documents by breaking them into manageable chunks
# - Preparing documents for semantic search and retrieval
# - Handling various document formats for AI processing

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_document_loading_splitting():
    print("\n--- Document Loading & Splitting Demo ---")
    
    # Load document content (using in-memory text for this example)
    # In real applications, you would typically load from files using DocumentLoaders
    print("\nLoading travel guide document...")

    travel_guide = """
        ## Tokyo
        Tokyo is Japan's bustling capital, mixing ultramodern and traditional elements. Visit the iconic Shibuya Crossing and Tokyo Skytree for city views. Don't miss the historic Senso-ji Temple in Asakusa or the peaceful Meiji Shrine gardens. For technology enthusiasts, Akihabara offers the latest gadgets and anime merchandise. Tokyo's food scene ranges from Michelin-starred restaurants to street food in Tsukiji Outer Market.

        ## Kyoto
        Kyoto, the cultural heart of Japan, is home to over 1,600 Buddhist temples and 400 Shinto shrines. The famous Golden Pavilion (Kinkaku-ji) and bamboo groves of Arashiyama are must-sees. Experience traditional tea ceremonies or spot geisha in the Gion district. Kyoto is particularly beautiful during cherry blossom season (late March to early April) and autumn foliage (November).

        ## Okinawa
        Okinawa is a Japanese island in the East China Sea. In Naha city, Shuri Castle is the rebuilt palace of the Ryukyu Kingdom. One of several remaining Ryukyuan fortresses on Okinawa from the Gusuku period, it features the ornate gate of Shureimon. The Okinawa Prefectural Museum has exhibitions on Okinawa’s natural and cultural heritage, plus a collection of fine art. Kokusai Street is lined with shops and restaurants.
    """

    print(f"Document loaded. Size: {len(travel_guide)} characters")
    
    # Split the document into chunks
    print("\nSplitting document into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Characters per chunk
        chunk_overlap=50,  # Overlap between chunks to maintain context
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],  # Priority order of separators
        length_function=len
    )
    
    # Create document objects from the text
    document_chunks = text_splitter.create_documents([travel_guide])
    print(f"Document successfully split into {len(document_chunks)} chunks")
    
    # Display example chunks to demonstrate the splitting
    print("\n--- Example Chunks ---")
    for i, chunk in enumerate(document_chunks[:10]):  # Show first ten chunks
        print(f"\nCHUNK {i+1}:")
        print(f"{chunk.page_content}")
        print("-" * 40)
    
    print("\nDocument splitting complete!")
    
    return document_chunks

def interactive_document_chat(document_chunks):
    print("\n--- Interactive Document Q&A ---")
    print("Ask questions about the Japan travel guide. Type 'exit' or 'quit' to end.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("Please enter a question.")
            continue
            
        # Simple approach: Find most relevant chunks by keyword matching
        # In real applications, you would use embeddings and semantic search
        relevant_chunks = []
        keywords = user_input.lower().split()
        for chunk in document_chunks:
            content = chunk.page_content.lower()
            if any(keyword in content for keyword in keywords):
                relevant_chunks.append(chunk.page_content)
        
        if not relevant_chunks:
            print("\nAI: I don't have specific information about that in my Japan travel guide.")
            continue
        
        # Prepare context and prompt
        context = "\n\n".join(relevant_chunks[:2])  # Limit to first 2 matches
        # print(f"\nContext: {context}")
        # print(f"\nUser Input: {user_input}")
        prompt = f"Based on this information about Japan:\n\n{context}\n\nAnswer this question: {user_input}"
        
        # Get response from LLM
        try:
            response = llm.invoke(prompt)
            print(f"\nAI: {response.content}")
        except Exception as e:
            print(f"\nAI: Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    document_chunks = run_document_loading_splitting()
    interactive_document_chat(document_chunks)
    print("\nDocument Loading & Splitting demo completed. Thank you for using the Travel Planning Assistant!")

# Travel Planning Assistant - Practical Example

"""
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load multiple travel guides from text files
print("Loading travel guides from directory...")
directory_loader = DirectoryLoader('./travel_guides/', glob="*.txt", loader_cls=TextLoader)
travel_documents = directory_loader.load()
print(f"Loaded {len(travel_documents)} travel guide documents")

# 2. Process documents with advanced splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n", "\. ", "! ", "\? ", ";", ",", " ", ""],
    length_function=len
)

split_documents = text_splitter.split_documents(travel_documents)
print(f"Split documents into {len(split_documents)} chunks")

# 3. Analyze document structure
destinations = {}
for chunk in split_documents:
    # Extract destination from metadata or content
    text = chunk.page_content.lower()
    for location in ["paris", "rome", "tokyo", "new york", "bali"]:
        if location in text:
            if location not in destinations:
                destinations[location] = 0
            destinations[location] += 1

print("\nDestination coverage in the travel guides:")
for destination, count in sorted(destinations.items(), key=lambda x: x[1], reverse=True):
    print(f"- {destination.title()}: {count} chunks")
"""

# Summary: This file demonstrates how to load documents and split them into manageable chunks
# for processing with language models. It shows how to configure the text splitter with different
# parameters to optimize chunk size and overlap. The example uses a Japan travel guide to show
# how document splitting works and includes a simple interactive Q&A system to demonstrate
# retrieving relevant chunks based on user questions.
