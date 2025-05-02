# chatbot.py (Improved Sentiment Handling with Question Detection, History, and Custom Responses)

import json
DEBUG = False # Set to True for verbose output, False for normal operation

import random
import numpy as np

import nltk
# Keep NLTK tokenization/stemming for custom response key generation
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.lancaster import LancasterStemmer

import os
from datetime import datetime

# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch

# --- Configuration ---
CONVERSATION_LOG_FILE = "conversation_log.txt"
CUSTOM_RESPONSES_FILE = "responses.json"
# DEBUG is controlled at the top

# --- Conversation History Configuration ---
MAX_HISTORY_TOKENS = 768
MAX_GEN_TOKENS = 200
NUM_RETURN_SEQUENCES = 1
TEMPERATURE = 0.6
TOP_K = 40
TOP_P = 0.9

# --- Sentiment Configuration ---
SENTIMENT_THRESHOLD = 0.85  # Increased confidence threshold (from first code block)
NEUTRAL_LABELS = {'neutral', 'NEUTRAL'}  # For question detection (from first code block)

# --- Initialize NLP tools ---
stemmer = LancasterStemmer()
nltk_tokenizer = TreebankWordTokenizer()

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)

# --- Load Models ---
print("Loading models...")
MODEL_NAME = "microsoft/DialoGPT-medium"

try:
    # Load DialoGPT (keep existing code)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.eval()

    # Updated sentiment analysis model with neutral detection (from first code block)
    sentiment_analyzer = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Sentiment analysis model loaded successfully")

except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Move model to device (keep existing code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Models loaded successfully and moved to device: {device}")

# --- Enhanced Sentiment Functions --- (from first code block)
def is_question(text):
    """Check if input is likely a straightforward question"""
    text_lower = text.lower()
    return any([
        text_lower.startswith(('what', 'when', 'where', 'which', 'who', 'why', 'how')),
        text_lower.endswith('?'),
        '?' in text_lower,
        any(word in text_lower.split() for word in {'are', 'is', 'do', 'does', 'did', 'can', 'can'}) # Added 'can' twice, corrected below
    ])

def is_question(text):
    """Check if input is likely a straightforward question"""
    text_lower = text.lower()
    question_starters = ('what', 'when', 'where', 'which', 'who', 'why', 'how')
    question_verbs = {'are', 'is', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'may', 'might'} # Added more common question verbs

    return any([
        text_lower.strip().endswith('?'),
        text_lower.strip().startswith(question_starters),
        any(word in text_lower.split() for word in question_verbs) and '?' in text_lower
    ])


def analyze_sentiment(text):
    """Analyze text sentiment with neutral detection (modified for new model)"""
    if not text or not sentiment_analyzer:
        return None
    
    try:
        # Use the new sentiment model
        result = sentiment_analyzer(text[:512]) # Truncate as before
        if result and isinstance(result, list):
             # Convert label to uppercase for consistency (from first code block)
            result[0]['label'] = result[0]['label'].upper()
            return result[0] # Return first result
    except Exception as e:
        if DEBUG: print(f"Debug: Sentiment analysis error: {e}")
    return None

def adjust_for_sentiment(generation_config, sentiment):
    """Adjust generation parameters based on sentiment (modified for new threshold)"""
    if sentiment and sentiment['score'] > SENTIMENT_THRESHOLD:
        label = sentiment['label']
        if label == 'NEGATIVE':
            generation_config.temperature = max(0.3, generation_config.temperature - 0.2)
            generation_config.top_p = 0.8
            if DEBUG: print("Debug: Adjusting for negative sentiment")
        elif label == 'POSITIVE':
            generation_config.temperature = min(0.9, generation_config.temperature + 0.1)
            if DEBUG: print("Debug: Adjusting for positive sentiment")
        # Note: Neutral sentiment doesn't explicitly adjust parameters based on the provided snippet
    return generation_config

def format_response_with_sentiment(response, sentiment, user_input):
    """Add sentiment prefixes only when appropriate (modified to check for questions)"""
    if is_question(user_input):
        return response  # No prefix for questions (from first code block)
    
    if sentiment and sentiment['score'] > SENTIMENT_THRESHOLD:
        label = sentiment['label']
        if label == 'NEGATIVE':
            return "I'm sorry to hear that. " + response
        elif label == 'POSITIVE':
            return "That's great! " + response
        # Note: No specific prefix for NEUTRAL sentiment based on the provided snippets
        
    return response # Return original response if no strong sentiment or if neutral


# --- Custom responses --- (keep existing code)
def load_custom_responses():
    """Loads custom responses from the JSON file."""
    if os.path.exists(CUSTOM_RESPONSES_FILE):
        try:
            with open(CUSTOM_RESPONSES_FILE, 'r', encoding='utf-8') as f:
                data = f.read().strip()
                return json.loads(data) if data else {}
        except json.JSONDecodeError:
            print(f"Warning: {CUSTOM_RESPONSES_FILE} contains invalid JSON. Starting with empty custom responses.")
            return {}
    return {}

def save_custom_response(key, response):
    """Saves a custom response to the JSON file."""
    data = load_custom_responses()
    data[key] = response
    try:
        with open(CUSTOM_RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving custom response: {e}")

def get_stemmed_key(text):
    """Generates a stemmed key for custom response lookup."""
    tokens = nltk_tokenizer.tokenize(text)
    stems = [stemmer.stem(w.lower()) for w in tokens]
    key = ' '.join(stems).strip()
    if not key:
        key = text.strip().lower()
    return key

# --- Logging --- (keep existing code)
def log_interaction(user_input, bot_response):
    """Logs the user input and bot response to a file."""
    try:
        with open(CONVERSATION_LOG_FILE, 'a', encoding='utf-8') as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] You: {user_input}\n")
            f.write(f"[{ts}] Bot: {bot_response}\n---\n")
    except Exception as e:
        if DEBUG: print(f"Debug: Error logging interaction: {e}")

# --- Chat loop --- (keep existing code structure and logic, integrate new sentiment/question handling)
def chat():
    print("Chatbot Ready! (Now with Enhanced Sentiment Awareness and Question Detection)")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Max history tokens: {MAX_HISTORY_TOKENS}")
    print(f"Sentiment threshold: {SENTIMENT_THRESHOLD}")
    print("Type 'quit' to exit.")

    custom_responses = load_custom_responses()
    if custom_responses:
        print(f"Loaded {len(custom_responses)} custom replies.")
    else:
        print("No custom replies loaded.")

    # Initialize conversation history
    conversation_history_ids = torch.tensor([[]], dtype=torch.long, device=device)
    conversation_attention_mask = torch.tensor([[]], dtype=torch.long, device=device)

    while True:
        try:
            inp = input("You: ")
            if inp.lower() == 'quit':
                print("Bot: Goodbye!")
                break

            user_input = inp.strip()
            if not user_input:
                continue

            # --- Step 1: Check Custom Responses ---
            stemmed_input_key = get_stemmed_key(user_input)
            if stemmed_input_key in custom_responses:
                bot_response = custom_responses[stemmed_input_key]
                print(f"Bot: {bot_response}")
                log_interaction(user_input, bot_response)

                # Update history (keep existing logic)
                user_input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
                bot_response_ids = tokenizer.encode(bot_response, return_tensors='pt').to(device)
                eos_token_id_tensor = torch.tensor([[tokenizer.eos_token_id]], device=device)

                new_history_segment_ids = torch.cat([
                    user_input_ids,
                    eos_token_id_tensor,
                    bot_response_ids,
                    eos_token_id_tensor
                ], dim=-1)

                conversation_history_ids = torch.cat([conversation_history_ids, new_history_segment_ids], dim=-1)
                conversation_attention_mask = torch.cat([
                    conversation_attention_mask,
                    torch.ones_like(new_history_segment_ids)
                ], dim=-1)

                if conversation_history_ids.shape[-1] > MAX_HISTORY_TOKENS:
                    conversation_history_ids = conversation_history_ids[:, -MAX_HISTORY_TOKENS:]
                    conversation_attention_mask = conversation_attention_mask[:, -MAX_HISTORY_TOKENS:]

                continue # Skip generation if custom response was used

            # --- Step 2: Analyze Sentiment --- (keep existing logic, use new function)
            sentiment = analyze_sentiment(user_input)
            if DEBUG and sentiment:
                print(f"Debug: Detected sentiment - {sentiment['label']} (confidence: {sentiment['score']:.2f})")

            # --- Step 3: Generate Response --- (keep existing logic, use new sentiment adjustment)
            user_input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
            eos_token_id_tensor = torch.tensor([[tokenizer.eos_token_id]], device=device)

            input_for_generation_ids = torch.cat([
                conversation_history_ids,
                user_input_ids,
                eos_token_id_tensor
            ], dim=-1)
            input_attention_mask = torch.ones_like(input_for_generation_ids)

            if input_for_generation_ids.shape[-1] > MAX_HISTORY_TOKENS:
                input_for_generation_ids = input_for_generation_ids[:, -MAX_HISTORY_TOKENS:]
                input_attention_mask = input_attention_mask[:, -MAX_HISTORY_TOKENS:]

            # Configure generation with sentiment awareness (use new adjustment function)
            generation_config = GenerationConfig(
                max_new_tokens=MAX_GEN_TOKENS,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            generation_config = adjust_for_sentiment(generation_config, sentiment)

            # Generate response (keep existing logic)
            output_sequence_ids = model.generate(
                input_for_generation_ids,
                attention_mask=input_attention_mask,
                generation_config=generation_config
            )

            # Process response (keep existing logic)
            prompt_length = input_for_generation_ids.shape[-1]
            if output_sequence_ids.shape[-1] > prompt_length:
                newly_generated_tokens_ids = output_sequence_ids[:, prompt_length:]
            else:
                newly_generated_tokens_ids = torch.tensor([[]], dtype=torch.long, device=device)

            decoded_response = tokenizer.decode(newly_generated_tokens_ids.squeeze(0), skip_special_tokens=False).strip()
            temp_response = decoded_response.rstrip()
            while temp_response.endswith(tokenizer.eos_token):
                temp_response = temp_response[:-len(tokenizer.eos_token)].rstrip()
            bot_response = temp_response if temp_response else "..."

            # Apply sentiment formatting (use the new function signature)
            # Modify the response formatting line as instructed:
            bot_response = format_response_with_sentiment(bot_response, sentiment, user_input)


            # Update history (keep existing logic)
            conversation_history_ids = output_sequence_ids
            conversation_attention_mask = torch.ones_like(conversation_history_ids)
            if conversation_history_ids.shape[-1] > MAX_HISTORY_TOKENS:
                conversation_history_ids = conversation_history_ids[:, -MAX_HISTORY_TOKENS:]
                conversation_attention_mask = conversation_attention_mask[:, -MAX_HISTORY_TOKENS:]

            print(f"Bot: {bot_response}")
            log_interaction(user_input, bot_response)

            # Learning prompt (keep existing logic)
            prompt_for_learning = "Would you like to teach me a specific response for that? (type your desired response, or 'skip')"
            print(f"Bot: {prompt_for_learning}")

            user_suggested_resp = input("Your suggested response (or 'skip'): ")
            if user_suggested_resp.strip().lower() not in ['skip', '']:
                save_custom_response(stemmed_input_key, user_suggested_resp.strip())
                custom_responses[stemmed_input_key] = user_suggested_resp.strip()
                print("Bot: Got it! I'll remember that.")

        except Exception as e:
            print(f"An error occurred: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()

# --- Main execution --- (keep existing code)
if __name__ == '__main__':
    # Validate custom responses file
    if not os.path.exists(CUSTOM_RESPONSES_FILE):
        try:
            with open(CUSTOM_RESPONSES_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not initialize custom responses file: {e}")
    else:
        try:
            with open(CUSTOM_RESPONSES_FILE, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {CUSTOM_RESPONSES_FILE} contains invalid JSON.")
            exit()

    chat()