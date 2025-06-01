"""
# Use Case: Zero-Shot Prompting
# - Translation
# - Factual Q&A
# - Summarization
# - Sentiment Classification
# - Entity Extraction
"""
import os
import openai
from dotenv import load_dotenv

def get_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    return openai.OpenAI(api_key=api_key)

def get_response(client, prompt, model="gpt-4o-mini", temperature=0.5):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def main():
    client = get_openai_client()
    prompts = [
        "Translate 'Hello, how are you?' into French.",
        "What is the capital of Japan?",
        "Summarize this paragraph: Artificial Intelligence is transforming industries by automating tasks, improving decision-making, and enhancing user experiences.",
        "Classify the sentiment of this sentence: 'I absolutely love this product!'",
        "Extract names from this text: 'Alice and Bob went to the market and met Charlie.'"
    ]
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response = get_response(client, prompt)
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()

"""
Summary:
    - Demonstrates how to make zero-shot calls to the OpenAI LLM using simple prompts
    - Shows how to run multiple prompt tasks in a loop
"""
