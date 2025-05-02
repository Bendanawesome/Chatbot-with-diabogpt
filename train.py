# train.py (Generative Chatbot Setup Script)

import os
import json
import sys

# Configuration
CUSTOM_RESPONSES_FILE = "responses.json"
CONVERSATION_LOG_FILE = "conversation_log.txt"

def initialize_file(filepath, default_content, is_json=False):
    """Helper function to initialize files with proper error handling"""
    try:
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                if is_json:
                    json.dump(default_content, f, indent=4)
                else:
                    f.write(default_content)
            print(f"Created new file: {filepath}")
            return True
        else:
            # Validate existing file if it's JSON
            if is_json:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json.load(f)
                    print(f"Validated existing file: {filepath}")
                except json.JSONDecodeError:
                    print(f"Error: {filepath} contains invalid JSON.", file=sys.stderr)
                    return False
            else:
                print(f"File already exists: {filepath}")
            return True
    except Exception as e:
        print(f"Error handling {filepath}: {e}", file=sys.stderr)
        return False

def main():
    print("\n--- Generative Chatbot Setup ---")
    print("This script prepares necessary files for the chatbot.")
    print("Note: The chatbot uses a pre-trained model (like DialoGPT).")
    print("----------------------------------\n")

    # Initialize all required files
    files_to_init = [
        (CUSTOM_RESPONSES_FILE, {}, True),  # (filename, default_content, is_json)
        (CONVERSATION_LOG_FILE, "", False)
    ]

    all_success = True
    for filepath, default_content, is_json in files_to_init:
        if not initialize_file(filepath, default_content, is_json):
            all_success = False

    if all_success:
        print("\nSetup completed successfully. You can now run chatbot.py")
    else:
        print("\nSetup completed with some errors. Please check the messages above.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()