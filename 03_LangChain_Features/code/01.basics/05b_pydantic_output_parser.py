# Use Case: Pydantic Output Parser
# - Extracting structured data from LLM responses in a type-safe way
# - Converting text outputs into structured Python objects
# - Creating APIs with consistent, validated response formats
# - Building applications that need specific structured data formats

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

def demonstrate_pydantic_parser():
    # Set up the Pydantic parser
    pydantic_parser = PydanticOutputParser(pydantic_object=ConceptDefinition)
    format_instructions = pydantic_parser.get_format_instructions()
    
    # Create the prompt template
    prompt_template = "Define the concept '{concept}'.\n{format_instructions}"
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["concept"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Create the chain
    concept_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain
    concept = "Vector Database"
    print("\n--- Pydantic Parser Demo ---")
    print(f"Concept to define: {concept}")
    print(f"Format Instructions: {format_instructions}\n")
    
    result = concept_chain.invoke({"concept": concept})
    output_text = result['text']
    
    print(f"Raw Output from LLM:\n{output_text}\n")
    
    # Parse the result
    parsed_output = pydantic_parser.parse(output_text)
    
    # Display the structured data
    print("Parsed Structured Output:")
    print(f"  Concept: {parsed_output.concept_name}")
    print(f"  Definition: {parsed_output.definition}")
    print("  Related Terms:")
    for term in parsed_output.related_terms:
        print(f"    - {term}")
    if parsed_output.example_use_case:
        print(f"  Example Use Case: {parsed_output.example_use_case}")
    print()
    
    # Demonstrate accessing as a Python object
    print("We can now access this data programmatically:")
    print(f"  parsed_output.concept_name: {parsed_output.concept_name}")
    print(f"  parsed_output.related_terms[0]: {parsed_output.related_terms[0]}")
    print(f"  Type of parsed_output: {type(parsed_output)}\n")


# Define the data structure for concept definitions
class ConceptDefinition(BaseModel):
    concept_name: str = Field(description="The name of the concept")
    definition: str = Field(description="A concise definition of the concept")
    related_terms: List[str] = Field(description="A list of related terms")
    example_use_case: Optional[str] = Field(description="An example use case of the concept")



def interactive_pydantic_chat():
    # Set up the parser
    product_parser = PydanticOutputParser(pydantic_object=ProductInfo)
    format_instructions = product_parser.get_format_instructions()
    
    print("\n--- Interactive Pydantic Parser Chat ---")
    print("Type 'exit' or 'quit' to end the chat.")
    print("Ask me to generate product information, and I'll return structured data.\n")
    print("Example: 'Create a product spec for a smart water bottle'\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("AI: Please enter some text.")
            continue
        
        # Create a prompt with the user's input
        prompt_text = f"{user_input}\n{format_instructions}"
        
        # Get response from LLM
        response = llm.invoke(prompt_text)
        output_text = response.content
        
        # Try to parse as structured data
        try:
            parsed_product = product_parser.parse(output_text)
            
            # Display in a user-friendly format
            print(f"\nAI: Here's the structured product information:\n")
            print(f"Product: {parsed_product.name}")
            print(f"Description: {parsed_product.description}")
            print("Key Features:")
            for feature in parsed_product.features:
                print(f"  - {feature}")
            print(f"Target Audience: {parsed_product.target_audience}")
            print(f"Price Range: {parsed_product.price_range}\n")
            
        except Exception:
            # If parsing fails, just show the raw output
            print(f"AI: {output_text}\n")
            print("(Note: Couldn't parse this as a product specification)\n")


# Define a product data model for the interactive chat
class ProductInfo(BaseModel):
    name: str = Field(description="Name of the product")
    description: str = Field(description="Brief description of the product")
    features: List[str] = Field(description="List of key features")
    target_audience: str = Field(description="The intended audience for this product")
    price_range: str = Field(description="Approximate price range for the product")


if __name__ == "__main__":
    demonstrate_pydantic_parser()
    interactive_pydantic_chat()
    print("Chat session ended. Thank you for using the Pydantic output parser demo!")



# Travel Planning Assistant - Practical Examples
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define a structured model for hotel recommendations
class HotelRecommendation(BaseModel):
    hotel_name: str = Field(description="The name of the recommended hotel")
    location: str = Field(description="The location/area within the city where the hotel is located")
    price_range: str = Field(description="Price range in USD, e.g., $100-200 per night")
    rating: float = Field(description="Hotel rating from 1-5 stars")
    amenities: List[str] = Field(description="List of notable amenities offered by the hotel")
    best_for: str = Field(description="Type of travelers this hotel is best suited for")
    walking_distance: List[str] = Field(description="Notable attractions or areas within walking distance")
    why_recommended: str = Field(description="Brief explanation of why this hotel is recommended for the traveler")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Initialize the parser
parser = PydanticOutputParser(pydantic_object=HotelRecommendation)
format_instructions = parser.get_format_instructions()

# Create the prompt template
hotel_template = ```Recommend a hotel in {city} for a traveler with the following preferences:
- Budget: {budget} per night
- Interests: {interests}
- Duration: {duration}
- Travelers: {travelers}

Provide a detailed recommendation for ONE hotel that meets these criteria.
{format_instructions}```

hotel_prompt = PromptTemplate(
    template=hotel_template,
    input_variables=["city", "budget", "interests", "duration", "travelers"],
    partial_variables={"format_instructions": format_instructions}
)

# Create the chain
hotel_chain = LLMChain(llm=llm, prompt=hotel_prompt)

# Generate a recommendation
result = hotel_chain.invoke({
    "city": "Barcelona", 
    "budget": "$150-250", 
    "interests": "architecture, local cuisine, and beach access", 
    "duration": "4 days",
    "travelers": "couple"
})

# Parse the output into a structured object
hotel_recommendation = parser.parse(result["text"])

# Use the structured data in your application
print(f"🏨 Recommended Hotel: {hotel_recommendation.hotel_name}")
print(f"📍 Location: {hotel_recommendation.location}")
print(f"💰 Price Range: {hotel_recommendation.price_range}")
print(f"⭐ Rating: {hotel_recommendation.rating}/5")
print(f"✨ Amenities: {', '.join(hotel_recommendation.amenities)}")
print(f"👥 Best For: {hotel_recommendation.best_for}")
print(f"🚶 Within Walking Distance: {', '.join(hotel_recommendation.walking_distance)}")
print(f"💡 Why We Recommend It: {hotel_recommendation.why_recommended}")

# Example of how you could use this structured data in a real application
def create_hotel_map_link(hotel_name, city):
    ```Generate a Google Maps link for the hotel```
    query = f"{hotel_name}, {city}".replace(" ", "+")
    return f"https://www.google.com/maps/search/?api=1&query={query}"

map_link = create_hotel_map_link(hotel_recommendation.hotel_name, "Barcelona")
print(f"\n🗺️ View on Map: {map_link}")
"""


# Summary: This file demonstrates how to use LangChain's PydanticOutputParser to extract structured
# data from LLM responses. It shows how to define Pydantic models for desired outputs, incorporate
# format instructions into prompts, and parse responses into strongly typed Python objects. The
# interactive chat showcases a scenario where the model handles schema-based parsing.