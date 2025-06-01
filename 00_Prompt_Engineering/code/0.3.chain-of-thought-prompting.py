"""
# Use Case: Chain-of-Thought (CoT) Prompting
# - Math problem solving (step-by-step)
# - Logic/reasoning tasks (step-by-step)
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
    cot_prompts = [
        {
            "title": "Step-by-Step Math Reasoning",
            "prompt": '''Solve the following math problem step by step.\n\nProblem: If a farmer has 3 fields, and each field produces 250 apples, how many apples does the farmer have in total?\n\nThink step by step before answering.'''
        },
        {
            "title": "Step-by-Step Logic Reasoning",
            "prompt": '''John, Mary, and Alex are in a race.\n- John finishes before Alex.\n- Mary finishes after John but before Alex.\n\nWho finishes last?\n\nThink step by step before answering.'''
        }
    ]
    for i, item in enumerate(cot_prompts, 1):
        print(f"\nExample {i}: {item['title']}")
        print(f"Prompt:\n{item['prompt']}")
        response = get_response(client, item['prompt'])
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()

"""
Summary:
    - Demonstrates step-by-step reasoning with LLMs using CoT prompts
    - Shows how to structure prompts for logical and mathematical reasoning
"""
