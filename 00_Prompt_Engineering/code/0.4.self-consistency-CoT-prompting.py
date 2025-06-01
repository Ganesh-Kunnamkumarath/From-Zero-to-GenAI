"""
# Use Case: Self-Consistency Chain-of-Thought (CoT)
# - Math problem solving with self-consistency
# - Any task benefitting from majority voting among LLM outputs
"""
import os
import openai
from collections import Counter
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
    prompt = (
        "Solve the following math problem step by step.\n\n"
        "Problem: A bakery sells 12 cupcakes per box. If a customer buys 5 boxes, how many cupcakes does the customer have in total?\n\n"
        "Think step by step before answering."
    )
    num_samples = 3  # Number of independent answers
    answers = []
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        answer = get_response(client, prompt)
        print(f"Response: {answer}")
        answers.append(answer)
    most_common_answer = Counter(answers).most_common(1)[0][0]
    print("\nAll Responses:")
    for idx, ans in enumerate(answers, 1):
        print(f"{idx}: {ans}")
    print(f"\nFinal Answer (Most Consistent): {most_common_answer}")

if __name__ == "__main__":
    main()

"""
Summary:
    - Demonstrates generating multiple step-by-step answers and selecting the most consistent one
    - Useful for tasks where LLMs may produce varied outputs
"""
