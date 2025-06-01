# Use Case: Conversation Memory
# - Creating chatbots that remember previous interactions
# - Building assistants that maintain context over multiple turns
# - Enabling follow-up questions without repeating context
# - Developing conversational AI with persistent memory

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("LLM Initialized successfully!\n")

def demonstrate_memory_chain():
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Where to store the memory in the prompt
        return_messages=True  # Return memory as a list of messages
    )
    
    # Define a prompt template with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that specializes in technology topics."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Create a chain with memory
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    print("\n--- Memory Chain Demo ---")
    
    # First interaction
    print("\nInteraction 1:")
    result1 = conversation_chain.invoke({"question": "What is LangChain?, Explain in two lines"})
    print(f"Response: {result1['text']}\n")
    
    # Second interaction - the model remembers the previous question
    print("Interaction 2:")
    result2 = conversation_chain.invoke({"question": "What are its main components?, Just list it, no explanation"})
    print(f"Response: {result2['text']}\n")
    
    # View the memory
    print("Memory Contents:")
    memory_variables = memory.load_memory_variables({})
    for i, message in enumerate(memory_variables["chat_history"]):
        print(f"[{message.type}] {message.content}")
    print()

def interactive_memory_chat():
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Define a prompt template with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Maintain a friendly, conversational tone."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Create a chain with memory
    memory_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    
    print("\n--- Interactive Memory Chat ---")
    print("Type 'exit' or 'quit' to end the chat.")
    print("Type 'memory' to see the current conversation history.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("AI: Please enter some text.")
            continue
        if user_input.lower() == "memory":
            # Show the current memory
            memory_vars = memory.load_memory_variables({})
            print("\n--- Current Conversation History ---")
            for i, message in enumerate(memory_vars["chat_history"]):
                print(f"[{message.type}] {message.content}")
            print()
            continue
        
        # Process through the memory chain
        result = memory_chain.invoke({"question": user_input})
        print(f"AI: {result['text']}\n")

if __name__ == "__main__":
    demonstrate_memory_chain()
    interactive_memory_chat()
    print("Chat session ended. Thank you for using the conversation memory demo!")

# Travel Planning Assistant - Practical Examples
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create a prompt template with travel advisor instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel advisor specializing in vacation planning. Provide thoughtful, "
              "personalized advice based on the traveler's preferences and previous questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Create the chain with memory
travel_advisor = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Example conversation - the assistant will remember details from previous messages
print("--- First interaction ---")
result1 = travel_advisor.invoke({"question": "I'm planning a trip to Southeast Asia for about 2 weeks. Which countries would you recommend?"})
print(f"AI: {result1['text']}\n")

print("--- Second interaction (with memory) ---")
result2 = travel_advisor.invoke({"question": "I'm particularly interested in food experiences. Which of these countries has the best cuisine?"})
print(f"AI: {result2['text']}\n")

print("--- Third interaction (with more context) ---")
result3 = travel_advisor.invoke({"question": "I also love beaches. Can you recommend a good itinerary that combines food and beaches?"})
print(f"AI: {result3['text']}\n")

# View memory contents
print("Memory Contents:")
memory_variables = memory.load_memory_variables({})
for i, message in enumerate(memory_variables["chat_history"]):
    print(f"[{message.type}] {message.content}")
"""

"""
Welcome to the LLM Zoo, where AI models roam wild and free!, What animal represents the Conversation Memory in this zoo
"""


# Summary: This file demonstrates how to use LangChain's memory capabilities to create a chatbot
# that remembers previous interactions. It shows how to integrate ConversationBufferMemory with an
# LLMChain to maintain context across multiple turns of conversation, enabling more natural
# and context-aware responses. The interactive chat allows users to see how memory affects responses.
