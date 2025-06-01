# Use Case: Comma Separated List Output Parser
# - Extracting structured lists from LLM responses
# - Parsing outputs into Python lists for further processing
# - Ensuring consistent, machine-readable formats for list data
# - Building applications that need to extract multiple items from text

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
print("LLM Initialized successfully!\n")

def demonstrate_list_parser():
    # Initialize the comma-separated list parser
    list_parser = CommaSeparatedListOutputParser()
    
    # Get formatting instructions for the model
    format_instructions = list_parser.get_format_instructions()
    
    # Create a prompt template with formatting instructions
    prompt_template = "List 5 key components of the LangChain framework.\n{format_instructions}"
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Create and run the chain
    list_chain = LLMChain(llm=llm, prompt=prompt)
    
    print("\n--- List Parser Demo ---")
    print("Prompt: List 5 key components of the LangChain framework.")
    print(f"Format Instructions: {format_instructions}\n")
    
    # Run the chain and parse the output
    result = list_chain.invoke({})
    raw_output = result['text']
    parsed_list = list_parser.parse(raw_output)
    
    print(f"Raw Output from LLM:\n{raw_output}\n")
    print(f"Parsed List (Python list object):\n{parsed_list}\n")
    
    # Demonstrate accessing individual elements
    print("Accessing individual elements:")
    for i, item in enumerate(parsed_list, 1):
        print(f"  {i}. {item}")
    print()

def interactive_list_parser_chat():
    # Initialize the parser
    list_parser = CommaSeparatedListOutputParser()
    format_instructions = list_parser.get_format_instructions()
    
    print("\n--- Interactive List Parser Chat ---")
    print("Type 'exit' or 'quit' to end the chat.")
    print("Ask for lists of things, and I'll return them as structured data.\n")
    print("Example: 'List 3 best practices for prompt engineering'\n")
    print("Example: 'List 3 most fun animals in zoo'\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("AI: Please enter some text.")
            continue
        
        # Create a dynamic prompt with the user's input
        prompt_template = f"{user_input}\n{format_instructions}"
        
        # Get the response from the LLM
        response = llm.invoke(prompt_template)
        raw_output = response.content
        
        # Try to parse as a list
        try:
            parsed_list = list_parser.parse(raw_output)
            print(f"AI: Here's your list as structured data:\n")
            for i, item in enumerate(parsed_list, 1):
                print(f"  {i}. {item}")
            print()  
        except Exception:
            # If parsing fails, just show the raw output
            print(f"AI: {raw_output}\n")
            print("(Note: Couldn't parse this as a comma-separated list)\n")

if __name__ == "__main__":
    demonstrate_list_parser()
    interactive_list_parser_chat()
    print("Chat session ended. Thank you for using the comma-separated list parser demo!")



# Travel Planning Assistant - Practical Examples
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Initialize the parser
list_parser = CommaSeparatedListOutputParser()
format_instructions = list_parser.get_format_instructions()

# Create a prompt template for generating packing lists
packing_template = ```Create a packing list for a {duration} trip to {destination} during {season}. 
Consider the typical weather and planned activities: {activities}.
{format_instructions}```

packing_prompt = PromptTemplate(
    template=packing_template,
    input_variables=["duration", "destination", "season", "activities"],
    partial_variables={"format_instructions": format_instructions}
)

# Create the chain
packing_chain = LLMChain(llm=llm, prompt=packing_prompt)

# Generate a packing list
result = packing_chain.invoke({
    "duration": "7 days", 
    "destination": "Iceland", 
    "season": "winter", 
    "activities": "hiking, hot springs, and northern lights viewing"
})

# Parse the output into a list
packing_list = list_parser.parse(result["text"])

# Display the packing list
print("Essential Items for Your Iceland Winter Trip:")
for i, item in enumerate(packing_list, 1):
    print(f"{i}. {item}")

# Now you can easily process this list programmatically
winter_clothes = [item for item in packing_list if "coat" in item.lower() or "jacket" in item.lower() or "thermal" in item.lower()]
print(f"\nWinter Clothing Items: {winter_clothes}")
"""


# Summary: This file demonstrates how to use LangChain's CommaSeparatedListOutputParser to extract
# structured lists from LLM responses. It shows how to provide format instructions to the LLM,
# parse its output into a Python list, and access individual elements. The interactive chat allows
# users to request lists and see them returned in a structured format for easier processing.