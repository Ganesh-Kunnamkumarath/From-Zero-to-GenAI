# Use Case: Prompt Templates and LLMChain
# - Creating structured prompts with variable inputs
# - Building reusable prompt templates for consistent LLM queries
# - Chaining prompts and LLMs for structured outputs
# - Creating interactive prompt-based applications

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

def demonstrate_prompt_template():
    # Create a simple prompt template
    define_template = "Do a {lines} liner SWOT analysis for the term: {term}"
    define_prompt = PromptTemplate(input_variables=["term", "lines"], template=define_template)
    
    # Create a chain with the prompt template and LLM
    define_chain = LLMChain(llm=llm, prompt=define_prompt)
    
    # Example usage
    term = "Large Language Model"
    lines = 3
    
    print(f"\n--- Prompt Template Demo: SWOT Analysis ---")
    print(f"Template: {define_template}")
    print(f"Variables: term='{term}', lines={lines}\n")
    
    # Invoke the chain
    result = define_chain.invoke({"term": term, "lines": lines})
    print(f"Result: {result['text']}\n")

def interactive_prompt_chat():
    # Create a template for the interactive chat
    chat_template = PromptTemplate(
        input_variables=["query"],
        template="You are an expert on LangChain. The user asks: {query}. Provide a helpful answer."
    )
    
    # Create a chain
    chat_chain = LLMChain(llm=llm, prompt=chat_template)
    
    print("\n--- Interactive Prompt Template Chat ---")
    print("Type 'exit' or 'quit' to end the chat.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("AI: Please enter some text.")
            continue
        
        # Use the chain with the user input
        result = chat_chain.invoke({"query": user_input})
        print(f"AI: {result['text']}\n")

if __name__ == "__main__":
    demonstrate_prompt_template()
    interactive_prompt_chat()
    print("Chat session ended. Thank you for using the prompt templates demo!")



# Travel Planning Assistant - Practical Examples
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Create a reusable template for trip itineraries
itinerary_template = "Create a {days}-day itinerary for a trip to {destination}. \nThe travelers are {travelers} and they are interested in {interests}. \nTheir budget is {budget} for the entire trip."

# Create the prompt template
itinerary_prompt = PromptTemplate(
    input_variables=["days", "destination", "travelers", "interests", "budget"],
    template=itinerary_template
)

# Create the chain
itinerary_chain = LLMChain(llm=llm, prompt=itinerary_prompt)

# Generate itinerary for a family trip to Japan
result = itinerary_chain.invoke({
    "days": "5", 
    "destination": "Tokyo, Japan", 
    "travelers": "a family with two children (ages 8 and 10)",
    "interests": "anime, technology, and traditional culture",
    "budget": "$5,000"
})

print(result["text"])
"""

"""
Welcome to the LLM Zoo, where AI models roam wild and free!, What animal represents the Prompt templates in this zoo
"""


# Summary: This file demonstrates how to use LangChain's PromptTemplate and LLMChain to create
# structured, reusable prompts with variable inputs. It shows how to define templates,
# pass variables into them, and chain them with LLMs to create consistent outputs.
# The interactive chat shows how templates can be used in a conversation context.
