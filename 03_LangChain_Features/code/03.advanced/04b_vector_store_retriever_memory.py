# Use Case: Vector Store Retriever Memory
# - Building travel assistants that can remember and retrieve specific details from past conversations
# - Handling complex trip planning across multiple sessions with semantically relevant memory recall
# - Creating more natural travel advice conversations by recalling related context on demand

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate # Ensure PromptTemplate is imported

# Load environment variables at the module level
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_vector_store_retriever_memory():
    print("\n--- Advanced Travel Assistant with Vector Store Memory Demo ---")
    print("A travel assistant that can retrieve specific details from your travel history")
    
    # Step 1: Set up the embeddings model for vector-based retrieval
    print("\nInitializing embeddings model and vector store memory...")
    embeddings = OpenAIEmbeddings()
    
    # Step 2: Create the vector store for semantic memory storage
    try:
        # Initialize the vector store with some travel memories
        # These will be searchable by semantic similarity
        memory_texts = [
            "The traveler mentioned they visited Paris last summer and loved the Louvre Museum, but found the Eiffel Tower too crowded.",
            "The traveler prefers boutique hotels over large chain hotels when traveling to European destinations.",
            "The traveler enjoys trying local street food and authentic cuisine rather than tourist-oriented restaurants.",
            "The traveler mentioned they have an upcoming trip to Japan in April and are interested in seeing cherry blossoms.",
            "The traveler has a moderate budget of around $150-200 per night for accommodations and prefers locations close to public transportation.",
            "The traveler expressed interest in cultural and historical sites rather than shopping or beaches.",
            "The traveler mentioned they have dietary restrictions and are vegetarian.",
            "The traveler prefers to travel during shoulder seasons to avoid crowds and high prices."
        ]
        memory_vector_store = FAISS.from_texts(memory_texts, embeddings)
        print("Travel memory vector store created successfully!")
    except Exception as e:
        print(f"Error initializing FAISS for travel memory: {e}")
        return
    
    # Step 3: Set up the memory retriever
    print("Configuring semantic memory retrieval system...")
    memory_retriever = memory_vector_store.as_retriever(search_kwargs=dict(k=2))  # Retrieve top 2 relevant memories
    
    # Initialize VectorStoreRetrieverMemory
    vector_memory = VectorStoreRetrieverMemory(
        retriever=memory_retriever,
        memory_key="relevant_travel_history"  # This key will hold the retrieved memories
    )
    
    # Step 4: Create a travel advisor prompt template that incorporates retrieved memories
    travel_template = """
You are a personalized travel assistant with access to the traveler's previous conversations.

Relevant pieces from the traveler's history:
{relevant_travel_history}

(Only reference this information if relevant to the current question. Don't explicitly mention that you're
accessing their history unless they ask.)

Current conversation:
Traveler: {input}
Travel Assistant:"""
    
    travel_prompt = PromptTemplate(
        input_variables=["input", "relevant_travel_history"],
        template=travel_template
    )
    
    # Step 5: Create the travel assistant chain with vector memory
    print("Building the travel assistant with semantic memory retrieval...")
    travel_assistant = ConversationChain(
        llm=llm,
        prompt=travel_prompt,
        memory=vector_memory,
        verbose=False  # Set to True to see prompt and memory interactions
    )
    
    print("Travel assistant with vector memory initialized successfully!")
    print("----------------------------------------\n")
    
    # Step 6: Demonstrate the travel assistant with vector memory
    print("--- Travel Assistant Memory Retrieval Demonstration ---")
    
    # Example 1: This should retrieve information about their Paris visit
    print("\nTraveler: I'm thinking about going back to France. Any recommendations based on my previous trips?")
    response1 = travel_assistant.invoke(input="I'm thinking about going back to France. Any recommendations based on my previous trips?")
    print(f"Travel Assistant: {response1['response']}")
    
    # Example 2: This should retrieve information about their accommodation preferences
    print("\nTraveler: What kind of hotels should I look for in Europe?")
    response2 = travel_assistant.invoke(input="What kind of hotels should I look for in Europe?")
    print(f"Travel Assistant: {response2['response']}")
    
    # Example 3: This should retrieve information about their food preferences
    print("\nTraveler: I'm concerned about finding good food while traveling. Any advice?")
    response3 = travel_assistant.invoke(input="I'm concerned about finding good food while traveling. Any advice?")
    print(f"Travel Assistant: {response3['response']}")
    
    # Example 4: This should retrieve information about their upcoming Japan trip
    print("\nTraveler: When was I planning to visit Japan again?")
    response4 = travel_assistant.invoke(input="When was I planning to visit Japan again?")
    print(f"Travel Assistant: {response4['response']}")
    
    # Add a new memory to the vector store for future retrieval
    print("\nAdding new travel memory to the system...")
    new_memory = "The traveler mentioned they speak conversational Spanish and feel comfortable traveling in Spanish-speaking countries."
    memory_vector_store.add_texts([new_memory])
    print("New memory added successfully!")
    
    # Test retrieval of the new memory
    print("\nTraveler: Which countries might be easy for me to navigate based on languages I speak?")
    response5 = travel_assistant.invoke(input="Which countries might be easy for me to navigate based on languages I speak?")
    print(f"Travel Assistant: {response5['response']}")
    print("----------------------------------------\n")
    
    # Step 7: Interactive Travel Assistant with Vector Memory
    print("\n--- Interactive Travel Assistant with Memory ---")
    print("Ask travel questions and the assistant will retrieve relevant details from your history.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'add memory: [your memory]' to add a new travel preference to the system.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("Please enter a question or statement.")
            continue
        
        # Check if the user wants to add a new memory
        if user_input.lower().startswith("add memory:"):
            new_memory_text = user_input[11:].strip()  # Extract the memory text
            if new_memory_text:
                try:
                    memory_vector_store.add_texts([f"The traveler mentioned that {new_memory_text}"])
                    print("New travel preference added to your profile.")
                except Exception as e:
                    print(f"Error adding memory: {e}")
            else:
                print("Please provide some content for your memory.")
            continue
        
        try:
            response = travel_assistant.invoke(input=user_input)
            print(f"Travel Assistant: {response['response']}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Travel Assistant with Memory!")

if __name__ == "__main__":
    run_vector_store_retriever_memory()
    print("Vector Store Retriever Memory example finished.")
