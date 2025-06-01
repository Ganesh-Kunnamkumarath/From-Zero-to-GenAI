"""
# Use Case: Few-Shot Prompting
# - Sentiment Classification with examples
# - English to French Translation with examples
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
        {
            "title": "Few-Shot Sentiment Classification",
            "prompt": '''Classify the sentiment of the following movie reviews as 'Positive' or 'Negative'.\n\nExample 1:\nReview: "The movie was an absolute masterpiece. The storytelling was gripping, and the characters were unforgettable."\nSentiment: Positive\n\nExample 2:\nReview: "I regret watching this movie. It was too long, boring, and the acting was terrible."\nSentiment: Negative\n\nExample 3:\nReview: "The cinematography was stunning, but the plot was weak and predictable."\nSentiment:'''
        },
        {
            "title": "Few-Shot English to French Translation",
            "prompt": '''Translate the following sentences from English to French.\n\nExample 1:\nEnglish: "Good morning!"\nFrench: "Bonjour!"\n\nExample 2:\nEnglish: "How are you?"\nFrench: "Comment ça va?"\n\nExample 3:\nEnglish: "Where is the nearest train station?"\nFrench:'''
        }
    ]
    for i, item in enumerate(prompts, 1):
        print(f"\nExample {i}: {item['title']}")
        print(f"Prompt:\n{item['prompt']}")
        response = get_response(client, item['prompt'])
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()

"""
Summary:
    - Demonstrates how to provide examples in prompts for improved LLM accuracy
    - Shows few-shot learning for different NLP tasks
"""