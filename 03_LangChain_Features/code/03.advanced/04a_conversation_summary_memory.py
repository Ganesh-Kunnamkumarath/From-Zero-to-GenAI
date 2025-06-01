# Use Case: Conversation Summary Memory
# - Creating travel planning assistants that remember previous discussions
# - Helping travelers maintain context across multiple trip planning sessions
# - Summarizing previous travel preferences to provide personalized recommendations

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain # Using LLMChain as in the original example for this memory type
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate # For the LLMChain prompt

# Load environment variables at the module level
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_conversation_summary_memory():
    print("\n--- Travel Planner with Conversation Summary Memory Demo ---")
    print("A travel assistant that summarizes your preferences for better recommendations")
    
    # Step 1: Initialize the conversation summary memory
    print("\nSetting up a travel planning assistant with memory...")
    
    travel_summary_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",  # This key will hold the conversation summary 
        input_key="input",         # Matches the input variable for the chain
        return_messages=False      # Return string summary rather than message objects
    )
    
    # Step 2: Create a travel advisor prompt template that includes the summarized history
    print("Configuring the travel planning assistant...")
    
    travel_advisor_template = """
You are a helpful travel planning assistant. You help travelers plan their trips by providing 
personalized recommendations based on their preferences and travel history.

Current conversation summary with the traveler:
{chat_history}

Based on this history, provide relevant travel advice and recommendations.

Traveler: {input}
Travel Assistant:"""
    
    travel_advisor_prompt = ChatPromptTemplate.from_template(travel_advisor_template)
    
    # Step 3: Create the Travel Advisor chain with the conversation summary memory
    travel_advisor_chain = LLMChain(
        llm=llm,
        prompt=travel_advisor_prompt,
        memory=travel_summary_memory,
        verbose=False  # Set to True to see prompt and memory interaction
    )
    
    print("Travel planning assistant initialized successfully!")
    print("----------------------------------------\n")
    
    # Step 4: Demonstrate the travel assistant with memory
    print("--- Travel Planning Session Demonstration ---")
    
    # First interaction - Destination preference
    print("\nTraveler: I'm planning a trip to Japan in cherry blossom season.")
    response1 = travel_advisor_chain.invoke(input="I'm planning a trip to Japan in cherry blossom season.")
    print(f"Travel Assistant: {response1['text']}")
    
    # Second interaction - Budget information
    print("\nTraveler: My budget is about $3000 for a 10-day trip, not including flights.")
    response2 = travel_advisor_chain.invoke(input="My budget is about $3000 for a 10-day trip, not including flights.")
    print(f"Travel Assistant: {response2['text']}")
    
    # Third interaction - Activity preferences
    print("\nTraveler: I'm interested in historical sites and local cuisine, but not too interested in shopping.")
    response3 = travel_advisor_chain.invoke(input="I'm interested in historical sites and local cuisine, but not too interested in shopping.")
    print(f"Travel Assistant: {response3['text']}")
    
    # Fourth interaction - Testing memory
    print("\nTraveler: Based on my preferences, what are your top 3 recommendations for places to visit in Japan?")
    response4 = travel_advisor_chain.invoke(input="Based on my preferences, what are your top 3 recommendations for places to visit in Japan?")
    print(f"Travel Assistant: {response4['text']}")
    
    # Check the memory contents to see how the conversation was summarized
    memory_vars = travel_summary_memory.load_memory_variables({})
    print("\nTravel Planning Summary (stored in memory):")
    print(memory_vars.get('chat_history', 'No summary found.'))
    print("----------------------------------------\n")
    
    # Step 5: Interactive Travel Planning Assistant
    print("\n--- Interactive Travel Planning Assistant ---")
    print("Continue the conversation with the travel assistant that remembers your preferences.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'memory' to see the current conversation summary.")
    
    # Create a new chain for interactive use with a fresh memory
    interactive_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="input"
    )
    
    interactive_chain = LLMChain(
        llm=llm,
        prompt=travel_advisor_prompt,
        memory=interactive_memory,
        verbose=False
    )
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("Please enter a question or statement.")
            continue
        if user_input.lower() == "memory":
            memory_content = interactive_memory.load_memory_variables({})
            print("\nCurrent Conversation Summary:")
            print(memory_content.get('chat_history', 'No conversation summary yet.'))
            continue
        
        try:
            response = interactive_chain.invoke(input=user_input)
            print(f"Travel Assistant: {response['text']}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Travel Planning Assistant!")

if __name__ == "__main__":
    run_conversation_summary_memory()
    print("Conversation Summary Memory example finished.")
