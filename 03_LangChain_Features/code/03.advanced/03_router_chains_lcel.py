# Use Case: Router Chains with LCEL
# - Directing travel queries to specialized handlers based on content
# - Creating a unified travel assistant that routes questions to the right expert
# - Providing different response types for different categories of travel questions

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate # Correct import for ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# Load environment variables at the module level
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_router_chain_lcel():
    print("\n--- Travel Question Router Demo ---")
    print("A smart travel assistant that routes queries to specialized experts")
    
    # Step 1: Define specialized prompt templates for different travel topics
    print("\nSetting up specialized travel experts...")
    
    # Template for accommodation questions
    accommodation_template = """You are an expert on travel accommodations worldwide. 
You provide detailed, helpful advice on hotels, hostels, resorts, Airbnbs, and other lodging options.
You always consider the traveler's preferences, budget constraints, and special requirements.

Here is a question about travel accommodations:
{input}"""
    
    # Template for transportation questions
    transportation_template = """You are a transportation and logistics expert for travelers.
You provide specific information about getting around in destinations, including public transit,
rental cars, flights, trains, and local transportation options.
You focus on efficiency, cost, and practical travel logistics.

Here is a question about travel transportation:
{input}"""
    
    # Template for cultural/sightseeing questions
    sightseeing_template = """You are a cultural expert and tour guide with extensive knowledge of attractions,
sightseeing opportunities, museums, historical sites, and local experiences worldwide.
You provide insightful recommendations that help travelers discover the most enriching experiences
at their destinations, both famous landmarks and hidden gems.

Here is a question about attractions and sightseeing:
{input}"""
    
    # Step 2: Create specialized chains using LangChain Expression Language (LCEL)
    print("Configuring specialized response chains...")
    
    # Create prompt objects
    accommodation_prompt = ChatPromptTemplate.from_template(accommodation_template)
    transportation_prompt = ChatPromptTemplate.from_template(transportation_template)
    sightseeing_prompt = ChatPromptTemplate.from_template(sightseeing_template)
    
    # Create specialized chains that will handle specific travel topics
    accommodation_chain = accommodation_prompt | llm | StrOutputParser()
    transportation_chain = transportation_prompt | llm | StrOutputParser()
    sightseeing_chain = sightseeing_prompt | llm | StrOutputParser()
    
    # Default chain for questions that don't fit specific categories
    default_prompt = ChatPromptTemplate.from_template(
        """You are a helpful general travel assistant. Provide useful information about travel-related questions.
        
Here's the travel question: {input}"""
    )
    default_chain = default_prompt | llm | StrOutputParser()
    
    # Step 3: Create the router prompt and classification chain
    print("Setting up the query classification system...")
    
    router_template = """Given the user's travel question below, classify it into exactly one of these categories: 
'accommodation', 'transportation', 'sightseeing', or 'general'.

- Use 'accommodation' for questions about hotels, hostels, resorts, where to stay, etc.
- Use 'transportation' for questions about flights, trains, rental cars, getting around, etc.
- Use 'sightseeing' for questions about attractions, tours, museums, things to do, etc.
- Use 'general' for other travel questions that don't fit the above categories.

Respond with ONLY the category name, nothing else.

Question: {input}

Category:"""
    
    router_prompt = ChatPromptTemplate.from_template(router_template)
    router_chain = router_prompt | llm | StrOutputParser()
    
    # Step 4: Create the RunnableBranch that will route questions to the right expert
    print("Building the routing decision system...")
    
    branch = RunnableBranch(
        (lambda x: "accommodation" in x["topic"].lower(), accommodation_chain),
        (lambda x: "transportation" in x["topic"].lower(), transportation_chain),
        (lambda x: "sightseeing" in x["topic"].lower(), sightseeing_chain),
        default_chain  # Default case for general travel questions
    )
    
    # Step 5: Create the full chain that first classifies the question, then routes to experts
    full_chain = {
        "topic": router_chain,
        "input": lambda x: x["input"]  # Pass original input through
    } | branch
    
    print("Travel Router Chain initialized successfully!")
    print("----------------------------------------\n")
    
    # Step 6: Test the Travel Router Chain with example questions
    print("--- Travel Router Chain Demonstration ---")
    
    # Test an accommodation question
    print("\nQuery 1: Accommodation Question")
    accommodation_question = "What are the best budget-friendly hotels in Tokyo for a family of four?"
    print(f"Q: {accommodation_question}")
    response_accommodation = full_chain.invoke({"input": accommodation_question})
    print(f"A: {response_accommodation}")
    
    # Test a transportation question
    print("\nQuery 2: Transportation Question")
    transportation_question = "What's the most efficient way to travel between Paris and Amsterdam?"
    print(f"Q: {transportation_question}")
    response_transportation = full_chain.invoke({"input": transportation_question})
    print(f"A: {response_transportation}")
    
    # Test a sightseeing question
    print("\nQuery 3: Sightseeing Question")
    sightseeing_question = "What are the must-see attractions in Bali for a first-time visitor?"
    print(f"Q: {sightseeing_question}")
    response_sightseeing = full_chain.invoke({"input": sightseeing_question})
    print(f"A: {response_sightseeing}")
    
    # Test a general travel question
    print("\nQuery 4: General Travel Question")
    general_question = "What's the best time of year to visit Australia?"
    print(f"Q: {general_question}")
    response_general = full_chain.invoke({"input": general_question})
    print(f"A: {response_general}")
    print("----------------------------------------\n")
    
    # Step 7: Interactive Travel Router Assistant
    print("\n--- Interactive Travel Router Assistant ---")
    print("Ask any travel question and see it routed to the appropriate expert.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_input = input("\nYour travel question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("Please enter a travel question.")
            continue
        
        try:
            # First get the classification to show the routing
            topic = router_chain.invoke({"input": user_input})
            print(f"Routing to: {topic.title()} Expert")
            
            # Then get the full response
            response = full_chain.invoke({"input": user_input})
            print(f"\nAnswer: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Travel Router Assistant!")

if __name__ == "__main__":
    run_router_chain_lcel()
    print("Router Chain (LCEL RunnableBranch) example finished.")
