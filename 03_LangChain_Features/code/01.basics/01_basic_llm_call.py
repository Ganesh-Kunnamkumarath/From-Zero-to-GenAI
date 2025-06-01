# Use Case: Basic LLM Interaction
# - Making simple direct calls to a language model
# - Creating an interactive chat interface with an LLM

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

def demonstrate_basic_call():
    # Demonstrate a single LLM call
    print("\n--- Basic LLM Call Demo ---")
    basic_prompt = "What is LangChain in one sentence?"
    response = llm.invoke(basic_prompt)
    print(f"Prompt: {basic_prompt}")
    print(f"Response: {response.content}\n")

def interactive_chat():
    # Start interactive chat mode
    print("\n--- Interactive Chat Mode ---")
    print("Type 'exit' or 'quit' to end the chat.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():  # Skip empty input
            print("AI: Please enter some text.")
            continue
        
        response = llm.invoke(user_input)
        print(f"AI: {response.content}\n")

if __name__ == "__main__":
    demonstrate_basic_call()
    interactive_chat()
    print("Chat session ended. Thank you for using the basic LLM chat demo!")




# Travel Planning Assistant - Practical Examples
"""
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Direct question to get travel recommendations
prompt = "What are 5 underrated beach destinations for a family vacation with young children?"
response = llm.invoke(prompt)
print(response.content)
"""


"""
Welcome to the LLM Zoo, where AI models roam wild and free!, What animal represents the basic LLM call in this zoo?
"""



# Summary: This file demonstrates the most basic way to interact with a language model using LangChain.
# It shows how to make a simple one-shot query to an LLM and how to create a simple interactive
# chat interface where users can have a conversation with the model.
