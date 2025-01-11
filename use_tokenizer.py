from pathlib import Path
from hindi_tokenizer import load_tokenizer, encode_text, decode_text

def main():
    # Load the trained tokenizer
    output_dir = Path("output")
    config_path = output_dir / "hindi_encoder.json"
    
    if not config_path.exists():
        print("Error: Tokenizer configuration not found! Please train the tokenizer first.")
        return
    
    tokenizer = load_tokenizer(str(config_path))
    
    # Interactive loop
    print("Hindi Text Encoder/Decoder (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        text = input("\nEnter Hindi text to encode/decode: ")
        
        if text.lower() == 'quit':
            break
        
        if not text.strip():
            continue
        
        # Encode the text
        token_ids, tokens = encode_text(tokenizer, text)
        print("\nEncoding:")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        
        # Decode back
        decoded_text = decode_text(tokenizer, token_ids)
        print("\nDecoding:")
        print(f"Text: {decoded_text}")

if __name__ == "__main__":
    main() 