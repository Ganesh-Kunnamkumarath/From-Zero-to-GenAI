# Use Case: Evaluation with QA Eval Chain
# - Assessing the quality of travel recommendation systems
# - Ensuring travel information accuracy before deployment
# - Comparing different travel information retrieval approaches

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

# Load environment variables at the module level
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please ensure you have the necessary API keys in your .env file")
    print("You can install dependencies with: python3 -m pip install -r requirements.txt")
    exit()

# Initialize the LLMs
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)  # For QA chain
eval_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # For evaluation (zero temperature)
print("LLMs Initialized successfully!\n")

def run_evaluation_qa_eval_chain():
    print("\n--- Travel Information Quality Evaluation Demo ---")
    print("A demonstration of how to evaluate travel information retrieval systems")
    
    # Step 1: Set up the embeddings model and travel knowledge base
    print("\nInitializing the travel knowledge base...")
    embeddings = OpenAIEmbeddings()
    
    # Create a comprehensive travel guide knowledge base
    travel_guide = """
## Travel Guide: Southeast Asia

### Thailand
Thailand is a popular destination known for its tropical beaches, opulent royal palaces, ancient ruins, and ornate temples. Bangkok, the capital, is a vibrant city with modern and traditional elements. The best time to visit Thailand is during the cool and dry season from November to early April. The Thai Baht (THB) is the local currency. Popular destinations include Bangkok, Chiang Mai, Phuket, Krabi, and Koh Samui. Thai cuisine is world-renowned for its balance of sweet, sour, salty, and spicy flavors, with dishes like Pad Thai and Tom Yum Goong being internationally famous.

### Vietnam
Vietnam offers diverse landscapes, from the terraced rice fields of Sapa to the limestone islands of Ha Long Bay. The country has a rich cultural heritage influenced by its history and neighbors. The Vietnamese Dong (VND) is the local currency. The best time to visit northern Vietnam is from October to December, while southern Vietnam is pleasant from November to April. Major destinations include Hanoi, Ho Chi Minh City, Ha Long Bay, Hoi An, and the Mekong Delta. Vietnamese cuisine features fresh ingredients, minimal use of oil, and flavor-enhancing herbs, with dishes like pho and banh mi being popular worldwide.

### Singapore
Singapore is a city-state known for its ultramodern architecture, vibrant food scene, and clean, safe environment. The Singapore Dollar (SGD) is the local currency. The weather is hot and humid year-round, with rainfall throughout the year. Major attractions include Marina Bay Sands, Gardens by the Bay, Sentosa Island, and the cultural neighborhoods of Chinatown and Little India. Singapore's cuisine reflects its multicultural heritage, with Chinese, Malay, Indian, and Western influences. Hawker centers offer affordable and delicious local dishes like Hainanese chicken rice and laksa.

### Indonesia
Indonesia is the world's largest archipelago, comprising over 17,000 islands. Bali is the most popular tourist destination, known for its beaches, volcanic mountains, and unique Hindu culture. The Indonesian Rupiah (IDR) is the local currency. The best time to visit is during the dry season from April to October. Beyond Bali, other notable destinations include Jakarta, Yogyakarta, Lombok, and the Komodo Islands. Indonesian cuisine varies by region but commonly features rice, noodles, and satay, with dishes like nasi goreng (fried rice) and rendang being staples.

## Transportation in Southeast Asia
Transportation options in Southeast Asia vary by country and region. Major cities typically have public transportation systems, while ride-sharing apps like Grab are widely available. For intercity travel, budget airlines such as AirAsia, Lion Air, and VietJet offer affordable flights. Trains are a good option in Thailand, Vietnam, and Malaysia, while long-distance buses connect most destinations. In islands like Bali and Phuket, renting a scooter is a popular way to get around, though traffic can be chaotic and international driving permits are often required.

## Visa Requirements
Visa requirements for Southeast Asian countries vary depending on the traveler's nationality. Many countries offer visa exemptions or visa-on-arrival for short stays (typically 30 days) for citizens of Western countries, Australia, and many Asian nations. Thailand, for instance, offers 30-day visa exemptions for citizens of over 50 countries. Singapore allows visa-free entry for most Western and Asian nationals. Vietnam requires e-visas for most visitors, while Indonesia offers visa-free entry or visa-on-arrival for many nationalities. Always check the latest visa requirements before travel, as policies can change.

## Cultural Etiquette
Respecting local customs and traditions is important when traveling in Southeast Asia. In Thailand and other Buddhist countries, it's inappropriate to touch someone's head or point your feet at people or religious objects. Modest dress is required when visiting temples. In Muslim areas of Indonesia and Malaysia, dress conservatively and be aware of prayer times and Ramadan observances. Public displays of affection are generally frowned upon throughout the region. Remove shoes before entering homes and some businesses. The concept of saving face is important, so avoid public confrontations or criticisms.
"""
    
    # Split the travel guide into chunks for the knowledge base
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    travel_chunks = text_splitter.create_documents([travel_guide])
    
    # Create the vector store and retriever
    vector_store = FAISS.from_documents(travel_chunks, embeddings)
    travel_retriever = vector_store.as_retriever()
    print(f"Travel knowledge base created with {len(travel_chunks)} chunks")
    
    # Step 2: Create the travel QA system to be evaluated
    print("\nSetting up the travel information system for evaluation...")
    travel_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=travel_retriever,
        return_source_documents=False
    )
    print("Travel information system initialized!")
    print("----------------------------------------\n")
    
    # Step 3: Create evaluation examples with ground truth answers
    print("--- Travel Information Evaluation Process ---")
    print("\nPreparing evaluation examples with expected answers...")
    
    travel_eval_examples = [
        {
            "query": "What is the best time to visit Thailand?", 
            "answer": "The best time to visit Thailand is during the cool and dry season from November to early April."
        },
        {
            "query": "What currency is used in Singapore?", 
            "answer": "The Singapore Dollar (SGD) is the local currency in Singapore."
        },
        {
            "query": "Name a popular destination in Vietnam.", 
            "answer": "Popular destinations in Vietnam include Hanoi, Ho Chi Minh City, Ha Long Bay, Hoi An, and the Mekong Delta."
        },
        {
            "query": "What transportation options are available in Southeast Asia?",
            "answer": "Transportation options include public transit in major cities, ride-sharing apps like Grab, budget airlines, trains, long-distance buses, and scooter rentals on islands."
        },
        {
            "query": "What are some important cultural etiquette tips for Southeast Asia?",
            "answer": "Important cultural etiquette includes not touching people's heads, not pointing feet at people or religious objects, dressing modestly at temples, removing shoes when entering homes, and avoiding public confrontations to help people save face."
        }
    ]
    
    # Step 4: Generate predictions using the travel QA system
    print("Generating travel information responses for evaluation...")
    travel_predictions = []
    
    for example in travel_eval_examples:
        try:
            response = travel_qa.invoke({"query": example["query"]})
            travel_predictions.append({"query": example["query"], "result": response['result']})
            print(f"Processed query: {example['query']}")
        except Exception as e:
            print(f"Error generating prediction for query '{example['query']}': {e}")
            travel_predictions.append({"query": example["query"], "result": "Error during prediction."})
    
    print("\nAll travel information responses generated successfully!")
    
    # Step 5: Evaluate the predictions using QAEvalChain
    print("\nEvaluating travel information quality...")
    travel_eval_chain = QAEvalChain.from_llm(llm=eval_llm)
    
    # Run the evaluation
    eval_results = travel_eval_chain.evaluate(
        travel_eval_examples,
        travel_predictions,
        question_key="query",
        answer_key="answer", 
        prediction_key="result"
    )
    
    # Step 6: Display the evaluation results
    print("\nTravel Information Quality Evaluation Results:")
    correct_count = 0
    
    for i, result in enumerate(eval_results):
        grade = result.get('results', 'N/A')
        if grade == "CORRECT":
            correct_count += 1
        
        print(f"\nQuery {i+1}: {travel_eval_examples[i]['query']}")
        print(f"Expected Answer: {travel_eval_examples[i]['answer']}")
        print(f"System Response: {travel_predictions[i]['result']}")
        print(f"Evaluation: {grade}")
        print("-" * 50)
    
    # Calculate and display the accuracy score
    accuracy = (correct_count / len(travel_eval_examples)) * 100
    print(f"\nOverall Travel Information Accuracy: {accuracy:.1f}%")
    print(f"Correct Answers: {correct_count}/{len(travel_eval_examples)}")
    
    if accuracy >= 80:
        print("Quality Assessment: EXCELLENT - The travel information system is highly reliable.")
    elif accuracy >= 60:
        print("Quality Assessment: GOOD - The travel information system is reliable but has room for improvement.")
    else:
        print("Quality Assessment: NEEDS IMPROVEMENT - The travel information system requires significant enhancement.")
    
    print("----------------------------------------\n")
    
    # Step 7: Interactive travel information evaluation
    print("\n--- Interactive Travel Information Evaluation ---")
    print("Ask a travel question and evaluate the response yourself.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_input = input("\nYour travel question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            print("Please enter a question.")
            continue
        
        try:
            # Get response from the travel QA system
            response = travel_qa.invoke({"query": user_input})
            print(f"\nTravel Information System: {response['result']}")
            
            # Ask user to evaluate the response
            rating = input("\nHow would you rate this response? (good/fair/poor): ").lower()
            if rating in ["good", "fair", "poor"]:
                print(f"You rated the response as: {rating.upper()}")
                print("This feedback could be used to improve the system in a production environment.")
            else:
                print("Invalid rating. Please use 'good', 'fair', or 'poor'.")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for evaluating the Travel Information System!")

if __name__ == "__main__":
    run_evaluation_qa_eval_chain()
    print("Evaluation with QAEvalChain example finished.")
