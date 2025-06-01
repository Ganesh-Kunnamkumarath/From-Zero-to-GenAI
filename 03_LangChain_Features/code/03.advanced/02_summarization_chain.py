# Use Case: Summarization Chain
# - Condensing lengthy travel documents into concise summaries
# - Creating trip reports and travel guides from detailed information
# - Extracting key information from lengthy destination descriptions

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document # Correct import for Document

# Load environment variables at the module level
load_dotenv()

# Check for API key
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
print("LLM Initialized successfully!\n")

def run_summarization_chain():
    print("\n--- Travel Document Summarization Demo ---")
    
    # Step 1: Create a detailed travel guide that needs summarization
    print("\nPreparing a detailed travel guide for summarization...")
    
    detailed_travel_guide = """
# Comprehensive Travel Guide to Japan

## Introduction to Japan
Japan is an island country in East Asia, known for its unique blend of ancient traditions and cutting-edge technology. The country consists of four main islands - Honshu, Hokkaido, Kyushu, and Shikoku - along with thousands of smaller islands. Japan's geography is predominantly mountainous, with a climate ranging from cool temperate in the north to subtropical in the south. The country experiences four distinct seasons, each offering unique travel experiences from cherry blossoms in spring to vibrant autumn foliage. Japan has a population of approximately 126 million people, with the Greater Tokyo Area being the most populous metropolitan area in the world.

## Historical Overview
Japan's history spans thousands of years, from the ancient Jomon period (14,500 BCE to 300 BCE) to the modern era. The country went through periods of isolation, feudal rule under shoguns and samurai, rapid modernization during the Meiji Restoration, imperial expansion, defeat in World War II, and remarkable economic recovery. Each historical period has left its mark on Japanese culture, architecture, and social customs, creating a fascinating tapestry for travelers to explore.

## Major Destinations

### Tokyo
Japan's bustling capital city blends ultramodern and traditional elements. Key attractions include:
- Shibuya Crossing: One of the busiest pedestrian crossings in the world
- Tokyo Skytree: A 634-meter tall broadcasting and observation tower
- Senso-ji Temple: Tokyo's oldest and most significant Buddhist temple
- Meiji Shrine: A serene Shinto shrine surrounded by forest in the heart of the city
- Akihabara: The electronic and anime/manga cultural center
- Imperial Palace: The primary residence of the Emperor of Japan
- Harajuku: Famous for youth culture and fashion

### Kyoto
The cultural and historical heart of Japan with over 1,600 Buddhist temples and 400 Shinto shrines:
- Kinkaku-ji (Golden Pavilion): A Zen temple covered in gold leaf
- Fushimi Inari Shrine: Famous for its thousands of vermilion torii gates
- Arashiyama Bamboo Grove: A mesmerizing pathway through towering bamboo stalks
- Gion District: Traditional entertainment district known for geisha
- Kiyomizu-dera: A UNESCO World Heritage site with spectacular views

### Osaka
Japan's kitchen and a hub for food lovers:
- Dotonbori: Entertainment district with vibrant nightlife and food stalls
- Osaka Castle: A historic landmark surrounded by a moat and park
- Universal Studios Japan: Popular theme park with attractions for all ages
- Kuromon Ichiba Market: Lively market with fresh seafood and local delicacies

### Hiroshima
A city with a powerful message of peace:
- Peace Memorial Park and Museum: Commemorating the atomic bombing of 1945
- Itsukushima Shrine: Famous for its "floating" torii gate on Miyajima Island
- Hiroshima Castle: Reconstructed samurai castle also known as the Carp Castle

## Transportation
Japan boasts one of the world's most efficient transportation systems:
- Shinkansen (Bullet Train): High-speed rail network connecting major cities
- Japan Rail Pass: Cost-effective option for tourists planning to travel extensively
- Tokyo Metro and Subway Systems: Comprehensive urban networks in major cities
- IC Cards (Suica, PASMO, ICOCA): Rechargeable smart cards for convenient payment
- Rental Bicycles: Available in many tourist areas for local exploration

## Cuisine
Japanese cuisine is renowned worldwide for its quality, presentation, and variety:
- Sushi and Sashimi: Fresh raw fish with rice or served alone
- Ramen: Noodle soup dishes with regional variations
- Tempura: Lightly battered and deep-fried seafood and vegetables
- Kaiseki: Traditional multi-course dinner with artistic presentation
- Okonomiyaki: Savory pancake containing various ingredients
- Street Food: Takoyaki (octopus balls), Yakitori (grilled chicken skewers), Taiyaki (fish-shaped cakes)

## Cultural Etiquette
Understanding Japanese customs enhances the travel experience:
- Bowing: The traditional Japanese greeting shows respect
- Shoes: Remove shoes before entering homes, traditional ryokans, and some restaurants
- Onsen (Hot Springs): Follow proper bathing etiquette and note that tattoos may be prohibited
- Tipping: Not customary in Japan and can sometimes cause confusion
- Chopstick Etiquette: Avoid pointing with chopsticks or sticking them vertically in rice

## Seasonal Highlights
- Spring (March-May): Cherry blossom season, mild temperatures
- Summer (June-August): Festivals, fireworks, hot and humid weather
- Autumn (September-November): Colorful foliage, comfortable temperatures
- Winter (December-February): Snow festivals in Hokkaido, excellent skiing conditions

## Practical Information
- Currency: Japanese Yen (¥)
- Language: Japanese, with limited English in tourist areas
- Electricity: 100V, outlets with two flat pins
- Internet: Widely available Wi-Fi and rental pocket Wi-Fi options
- Emergency Number: 110 (Police), 119 (Fire/Ambulance)
- Visa Requirements: Varies by nationality, many countries eligible for visa-free short stays

## Accommodation Options
- Ryokan: Traditional Japanese inns with tatami floors and futon bedding
- Capsule Hotels: Compact sleeping pods popular in urban areas
- Business Hotels: Affordable, no-frills accommodation with basic amenities
- Luxury Hotels: International chains and Japanese luxury properties
- Minshuku: Family-run Japanese bed and breakfasts
- Temple Lodging (Shukubo): Overnight stays in Buddhist temples
"""
    
    # Step 2: Create Document object for the summarization chain
    travel_doc = Document(page_content=detailed_travel_guide)
    
    print("\nSetting up the summarization chain...")
    # Step 3: Load the summarization chain with different approaches
    
    # Map-reduce approach (good for longer documents)
    map_reduce_chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce",
        verbose=False
    )
    
    # Step 4: Generate summary
    print("\n--- Map-Reduce Summarization Results ---")
    print("Generating a concise travel guide summary...")
    map_reduce_result = map_reduce_chain.invoke({"input_documents": [travel_doc]})
    map_reduce_summary = map_reduce_result['output_text']
    
    print(f"\nOriginal Travel Guide Length: {len(detailed_travel_guide)} characters")
    print(f"Summary Length: {len(map_reduce_summary)} characters")
    print(f"Reduction: {(1 - len(map_reduce_summary) / len(detailed_travel_guide)) * 100:.1f}%")
    print("\nTravel Guide Summary:")
    print(f"\n{map_reduce_summary}")
    
    # Step 5: Now show an example with another approach - "stuff" (for shorter documents)
    print("\n--- Stuff Method Summarization Example ---")
    print("For comparison, generating a summary of just the Kyoto section...")
    
    kyoto_section = """
### Kyoto
The cultural and historical heart of Japan with over 1,600 Buddhist temples and 400 Shinto shrines:
- Kinkaku-ji (Golden Pavilion): A Zen temple covered in gold leaf
- Fushimi Inari Shrine: Famous for its thousands of vermilion torii gates
- Arashiyama Bamboo Grove: A mesmerizing pathway through towering bamboo stalks
- Gion District: Traditional entertainment district known for geisha
- Kiyomizu-dera: A UNESCO World Heritage site with spectacular views
"""
    
    kyoto_doc = Document(page_content=kyoto_section)
    stuff_chain = load_summarize_chain(llm=llm, chain_type="stuff")
    stuff_result = stuff_chain.invoke({"input_documents": [kyoto_doc]})
    
    print(f"\nKyoto Section Summary:\n{stuff_result['output_text']}")
    print("----------------------------------------\n")

if __name__ == "__main__":
    run_summarization_chain()
    print("Summarization Chain example finished.")

    print("\n--- Interactive Chat Mode ---")
    print("Type 'exit' or 'quit' to end the chat.")
    
    try:
        # ChatOpenAI is already imported.
        # The OPENAI_API_KEY should be loaded by run_summarization_chain()
        chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        print("Interactive LLM Initialized.")
    except Exception as e:
        print(f"Error initializing LLM for interactive mode: {e}")
        print("Please ensure your OPENAI_API_KEY is correctly set in the .env file.")
        chat_llm = None 

    if chat_llm: 
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip(): 
                print("AI: Please enter some text.")
                continue
            
            try:
                response = chat_llm.invoke(user_input)
                print(f"AI: {response.content}\n")
            except Exception as e:
                print(f"AI: Error during API call: {e}\n")

    print("Exiting interactive chat.")
