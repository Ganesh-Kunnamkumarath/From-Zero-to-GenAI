# Use Case: Response Streaming
# - Providing real-time responses to users instead of waiting for the complete response
# - Creating more responsive and interactive AI interfaces
# - Handling long responses in a way that feels more natural to users

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_streaming():
    print("\n--- Response Streaming Demo ---")
    
    # Create a streaming-enabled LLM (same model but with streaming turned on)
    print("\nInitializing streaming model...")
    streaming_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.5,
        streaming=True  # Explicitly enable streaming
    )
    print("Streaming model initialized successfully!")
    
    # Demonstrate streaming with a travel itinerary generation
    print("\nGenerating a travel itinerary in streaming mode (watch it appear word by word):\n")
    
    travel_prompt = "Create a very short 3-day itinerary for a first-time visitor to Tokyo, Japan. Include specific attractions, recommended restaurants, and travel tips."
    
    print(f"Prompt: {travel_prompt}\n")
    print("Response:\n")
    
    try:
        # Stream the response chunk by chunk
        for chunk in streaming_llm.stream([HumanMessage(content=travel_prompt)]):
            if hasattr(chunk, 'content'):
                print(chunk.content, end="", flush=True)
                time.sleep(0.04)  # Slight delay for better visual effect
    except Exception as e:
        print(f"\n\nAn error occurred during streaming: {e}")
    
    print("\n\nStreaming complete!")
    print("\nThis demonstrates how streaming can provide a more interactive experience")
    print("for users waiting for lengthy responses like travel itineraries.")

if __name__ == "__main__":
    run_streaming()
    print("\nStreaming demo completed. Thank you for using the Travel Planning Assistant!")


# Streamlit Example (not for direct execution)
'''
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st

# Create a Streamlit web interface for a streaming travel assistant
st.title("🧳 Real-Time Travel Assistant")

# Setup the streaming model
streaming_llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.7)

'''

# Summary: This file demonstrates how to use LangChain's streaming capabilities to provide
# real-time responses to users. Instead of waiting for the complete response to be generated,
# streaming delivers content as it's being created, word by word. This creates a more
# engaging and interactive experience, especially for longer responses like travel itineraries
# or destination descriptions where users would otherwise face a long wait time.
