import re
import requests
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np

class TrieNode:
    """Node in the prefix tree (trie) for fast token matching"""
    def __init__(self):
        self.children = {}
        self.is_token = False
        self.token = None

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.chars = []  # List of unique characters
        self.stoi = {}   # String to index mapping
        self.itos = {}   # Index to string mapping
        self.data = []   # Encoded text data
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        
        # Statistics tracking
        self.stats = {
            "vocab_sizes": [],
            "data_sizes": [],
            "compression_ratios": [],
            "merge_counts": [],
            "tokens_created": [],
            "max_token_lengths": [1],
        }
        
        self.original_length = 0
        self.max_token_length = 1
        
    def initialize_vocab(self, text):
        """Initialize vocabulary from characters in text"""
        # Preprocess text first
        text = preprocess_hindi_text(text)
        
        # Get unique characters and add special tokens
        chars = sorted(list(set(text)))
        all_tokens = self.special_tokens + chars
        
        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(all_tokens)}
        self.itos = {i: ch for i, ch in enumerate(all_tokens)}
        
        # Initial encoding
        self.data = [self.stoi[c] for c in text]
        self.original_length = len(self.data)
        
        # Initialize stats
        self.stats["vocab_sizes"].append(len(self.stoi))
        self.stats["data_sizes"].append(len(self.data))
        self.stats["compression_ratios"].append(1.0)
        
    def get_digram_stats(self):
        """Optimized digram counting using Counter"""
        # Pre-compute pairs for all data at once
        pairs = zip(self.data, self.data[1:])
        return Counter((int(pair[0]), int(pair[1])) for pair in pairs)
    
    def replace_byte_pair_in_data(self, pair, new_token):
        """Optimized pair replacement using numpy"""
        data = np.array(self.data)
        i = 0
        result = []
        
        # Use numpy's vectorized operations
        while i < len(data) - 1:
            if data[i] == pair[0] and data[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(data[i])
                i += 1
        
        if i == len(data) - 1:
            result.append(data[-1])
            
        return result
    
    def encode_pair(self, pair):
        """Add a new token to vocabulary from pair"""
        pair_str = self.itos[pair[0]] + self.itos[pair[1]]
        next_idx = len(self.itos)
        self.stoi[pair_str] = next_idx
        self.itos[next_idx] = pair_str
        
        # Update max token length
        self.max_token_length = max(self.max_token_length, len(pair_str))
        return next_idx
    
    def train(self, texts, min_frequency=2, print_interval=500):
        """Optimized BPE training with vectorized operations"""
        # Combine all texts and initialize vocab
        print("Initializing vocabulary...")
        full_text = " ".join(texts)
        self.initialize_vocab(full_text)
        
        # Convert data to numpy array for faster operations
        data = np.array(self.data, dtype=np.int32)
        
        # Pre-compute character frequencies using numpy
        print("Computing initial frequencies...")
        unique, counts = np.unique(data, return_counts=True)
        char_freqs = dict(zip(unique, counts))
        
        # Initialize progress bar
        pbar = tqdm(total=self.vocab_size - len(self.stoi), 
                   desc="Training BPE",
                   position=0)
        
        # Batch processing parameters
        batch_size = min(1000, self.vocab_size - len(self.stoi))
        stats_buffer = []
        
        while len(self.stoi) < self.vocab_size:
            # Get pair frequencies using vectorized operations
            # Create a view of consecutive pairs
            pair_view = np.lib.stride_tricks.sliding_window_view(data, 2)
            
            # Convert to tuples for counting
            pairs = [tuple(pair) for pair in pair_view]
            pair_counts = Counter(pairs)
            
            if not pair_counts:
                break
            
            # Get top pairs for batch processing
            top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:batch_size]
            
            # Process batch of pairs
            for (token1, token2), freq in top_pairs:
                if len(self.stoi) >= self.vocab_size:
                    break
                
                # Create new token
                new_idx = self.encode_pair((token1, token2))
                
                # Vectorized pair replacement
                # Create a boolean mask for matching pairs
                pair_mask = (data[:-1] == token1) & (data[1:] == token2)
                if not np.any(pair_mask):
                    continue
                
                # Create new data array efficiently
                indices = np.where(pair_mask)[0]
                new_data = np.empty(len(data) - len(indices), dtype=np.int32)
                
                # Fill new data array using vectorized operations
                pos = 0
                prev_idx = 0
                for idx in indices:
                    # Copy unchanged elements
                    new_data[pos:pos + (idx - prev_idx)] = data[prev_idx:idx]
                    pos += idx - prev_idx
                    # Add merged token
                    new_data[pos] = new_idx
                    pos += 1
                    prev_idx = idx + 2
                
                # Copy remaining elements
                if prev_idx < len(data):
                    new_data[pos:] = data[prev_idx:]
                
                data = new_data
                
                # Update statistics
                stats_buffer.append({
                    'vocab_size': len(self.stoi),
                    'data_size': len(data),
                    'merge_count': freq,
                    'new_token': self.itos[new_idx]
                })
                
                pbar.update(1)
                
                # Batch update statistics
                if len(stats_buffer) >= print_interval:
                    self._update_stats_batch(stats_buffer)
                    if print_interval:
                        self.print_progress(
                            len(self.stoi),
                            stats_buffer[-1]['new_token'],
                            stats_buffer[-1]['merge_count']
                        )
                    stats_buffer = []
        
        # Final statistics update
        if stats_buffer:
            self._update_stats_batch(stats_buffer)
        
        pbar.close()
        self.data = data.tolist()
        
        # Calculate final compression ratio
        final_ratio = self.original_length / len(self.data)
        print(f"\nTraining completed. Final vocabulary size: {len(self.stoi)}")
        print(f"Final compression ratio: {final_ratio:.2f}")
    
    def _update_stats_batch(self, stats_buffer):
        """Update statistics in batch for better performance"""
        if not stats_buffer:
            return
            
        # Update all statistics at once
        self.stats["vocab_sizes"].extend(s['vocab_size'] for s in stats_buffer)
        self.stats["data_sizes"].extend(s['data_size'] for s in stats_buffer)
        self.stats["merge_counts"].extend(s['merge_count'] for s in stats_buffer)
        self.stats["tokens_created"].extend(s['new_token'] for s in stats_buffer)
        
        # Update compression ratios
        new_ratios = [self.original_length / s['data_size'] for s in stats_buffer]
        self.stats["compression_ratios"].extend(new_ratios)
        
        # Update max token lengths
        self.stats["max_token_lengths"].extend([self.max_token_length] * len(stats_buffer))
    
    def print_progress(self, iteration, new_token, merge_count):
        """Print training progress"""
        print(f"\nIteration {iteration:,}")
        print(f"Created token: '{new_token}' (merged {merge_count:,} times)")
        print(f"Current vocabulary size: {len(self.stoi):,}")
        print(f"Current data size: {len(self.data):,}")
        print(f"Current compression ratio: {self.stats['compression_ratios'][-1]:.2f}")
        print("-" * 80)
    
    def plot_statistics(self):
        """Plot training statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Vocabulary Size vs Data Size
        ax1.plot(self.stats["vocab_sizes"], self.stats["data_sizes"])
        ax1.set_xlabel("Vocabulary Size")
        ax1.set_ylabel("Dataset Size")
        ax1.set_title("Vocabulary Size vs Dataset Size")
        
        # Plot 2: Compression Ratio vs Vocabulary Size
        ax2.plot(self.stats["vocab_sizes"], self.stats["compression_ratios"])
        ax2.set_xlabel("Vocabulary Size")
        ax2.set_ylabel("Compression Ratio")
        ax2.set_title("Compression Ratio vs Vocabulary Size")
        
        # Plot 3: Merge Counts Distribution
        if self.stats["merge_counts"]:
            ax3.hist(self.stats["merge_counts"], bins=30)
            ax3.set_xlabel("Number of Merges")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Distribution of Merge Counts")
        
        # Plot 4: Token Lengths Over Time
        if self.stats["tokens_created"]:
            token_lengths = [len(token) for token in self.stats["tokens_created"]]
            ax4.plot(range(len(token_lengths)), token_lengths)
            ax4.set_xlabel("Merge Operation")
            ax4.set_ylabel("New Token Length")
            ax4.set_title("Token Length Evolution")
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str) -> None:
        """Save tokenizer state to a JSON file"""
        state = {
            "stoi": self.stoi,
            "itos": self.itos,
            "max_token_length": self.max_token_length,
            "stats": self.stats,
            "special_tokens": self.special_tokens
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load tokenizer state from a JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # Create new instance
        instance = cls()
        
        # Convert string keys to integers in itos
        instance.itos = {int(k): v for k, v in state["itos"].items()}
        # Convert string values to integers in stoi
        instance.stoi = {k: int(v) for k, v in state["stoi"].items()}
        instance.max_token_length = state["max_token_length"]
        instance.stats = state["stats"]
        instance.special_tokens = state["special_tokens"]
        
        # Debug info
        print(f"Loaded vocabulary size: {len(instance.itos)}")
        print(f"Max token ID: {max(instance.itos.keys())}")
        print(f"Sample tokens: {list(instance.itos.items())[:5]}")
        
        return instance
    
    def encode(self, text: str):
        """Convert text to token indices"""
        # Preprocess input text
        text = preprocess_hindi_text(text)
        
        tokens = []
        token_ids = []
        
        # Split text into words
        words = text.split()
        
        for word in words:
            # Try to find longest matching token
            while word:
                longest_match = None
                for token, idx in sorted(self.stoi.items(), key=lambda x: len(x[0]), reverse=True):
                    if word.startswith(token):
                        longest_match = (token, idx)
                        break
                
                if longest_match:
                    token, idx = longest_match
                    tokens.append(token)
                    token_ids.append(idx)
                    word = word[len(token):]
                else:
                    # Skip unknown character and continue
                    word = word[1:]
        
        return token_ids, tokens
    
    def decode(self, token_ids: list) -> str:
        """Convert token indices back to text with better error handling"""
        decoded_tokens = []
        max_id = max(self.itos.keys())
        
        for idx in token_ids:
            try:
                # Convert to int and check range
                idx = int(idx) if isinstance(idx, str) else idx
                if idx < 0 or idx > max_id:
                    continue
                
                # Get token from vocabulary
                if idx in self.itos:
                    token = self.itos[idx]
                    if token not in self.special_tokens:
                        # Add token with space
                        decoded_tokens.append(token)
                
            except (ValueError, KeyError):
                continue
        
        # Join all tokens with spaces and clean up extra spaces
        result = " ".join(token for token in decoded_tokens if token.strip())
        # Remove duplicate spaces and strip
        result = " ".join(result.split())
        return result

def download_dataset(url, filepath, max_size_gb=2):
    """
    Downloads a portion of the dataset with size limit and resume capability.
    
    Args:
        url (str): URL of the dataset
        filepath (Path): Path where the file should be saved
        max_size_gb (float): Maximum size to download in gigabytes
    """
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    
    # Check if we already have enough data
    if filepath.exists() and filepath.stat().st_size >= max_size_bytes:
        print(f"Already have {max_size_gb}GB of data, skipping download.")
        return
    
    print(f"Downloading first {max_size_gb}GB from {url}")
    
    # Get the current size if file exists (for resume)
    current_size = filepath.stat().st_size if filepath.exists() else 0
    
    # Set up headers for resume
    headers = {'Range': f'bytes={current_size}-'} if current_size > 0 else {}
    
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = min(
            int(response.headers.get('content-length', 0)) + current_size,
            max_size_bytes
        )
        
        mode = 'ab' if current_size > 0 else 'wb'
        with open(filepath, mode) as file, tqdm(
            desc="Downloading",
            initial=current_size,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=8192):
                if not data:
                    break
                    
                size = file.write(data)
                progress_bar.update(size)
                
                # Check if we've reached the size limit
                if file.tell() >= max_size_bytes:
                    print(f"\nReached {max_size_gb}GB limit, stopping download.")
                    break
                    
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        if filepath.exists():
            print("Partial download remains available for resume.")
        raise

def prepare_dataset(input_path, sample_size=None, max_lines=None):
    """
    Prepares the dataset by optionally sampling and basic cleaning.
    
    Args:
        input_path (Path): Path to the raw dataset
        sample_size (int, optional): Number of lines to sample. If None, use entire dataset
        max_lines (int, optional): Maximum number of lines to read from file
    
    Returns:
        list: Processed lines from the dataset
    """
    print("Reading and preparing dataset...")
    lines = []
    
    with open(input_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file, desc="Reading lines")):
            if max_lines and i >= max_lines:
                break
                
            if line.strip():
                lines.append(line)
                if sample_size and len(lines) >= sample_size:
                    break
    
    return lines

def preprocess_hindi_text(text):
    """
    Preprocesses Hindi text by removing unwanted characters and normalizing punctuation.
    
    Args:
        text (str): Raw Hindi text input
    
    Returns:
        str: Cleaned and normalized text
    """
    # Remove <unk> tokens first
    text = text.replace("<unk>", "")
    
    # Retain Hindi characters and punctuation
    text = re.sub(r"[^\u0900-\u097F\s।,.!?\-]", "", text)
    # Remove digits (both English and Hindi)
    text = re.sub(r"[0-9०-९]", "", text)
    # Normalize full stops and whitespace
    text = re.sub(r"।", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_compression_ratio(tokenizer, corpus_path):
    """
    Calculates the compression ratio for the tokenizer on the given corpus.
    
    Args:
        tokenizer (Tokenizer): Trained BPE tokenizer
        corpus_path (str): Path to the preprocessed corpus
    
    Returns:
        float: Compression ratio (characters/tokens)
    """
    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
    
    total_chars = sum(len(line) for line in corpus)
    total_tokens = sum(len(tokenizer.encode(line).tokens) for line in corpus)
    return total_chars / total_tokens

def encode_text(tokenizer, text):
    cleaned_text = preprocess_hindi_text(text)
    return tokenizer.encode(cleaned_text)

def decode_text(tokenizer, token_ids):
    return tokenizer.decode(token_ids)

def test_tokenizer(tokenizer, test_text):
    """
    Tests the tokenizer by encoding and decoding sample text.
    
    Args:
        tokenizer (Tokenizer): Trained BPE tokenizer
        test_text (str): Sample text for testing
    """
    print("\nTokenizer Test:")
    print("-" * 50)
    print(f"Original Text: {test_text}")
    
    # Encode
    token_ids, tokens = encode_text(tokenizer, test_text)
    print(f"\nTokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_text = decode_text(tokenizer, token_ids)
    print(f"\nDecoded Text: {decoded_text}")

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset URL and paths
    dataset_url = "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt"
    raw_dataset_path = Path("raw_hindi_dataset.txt")
    preprocessed_path = output_dir / "preprocessed_hindi.txt"
    
    # Step 1: Download dataset if it doesn't exist or is too small
    if not raw_dataset_path.exists() or raw_dataset_path.stat().st_size < (10 * 1024 * 1024 * 1024):
        print("Step 1: Downloading dataset (10GB limit)...")
        try:
            download_dataset(dataset_url, raw_dataset_path, max_size_gb=10)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            if not raw_dataset_path.exists():
                return
            print("Continuing with existing partial download...")
    else:
        print("Sufficient dataset already exists, skipping download.")
    
    # Step 2: Prepare and preprocess the dataset
    print("Step 2: Preprocessing dataset...")
    try:
        # Sample 2 Million lines from the first 3 Million lines
        raw_data = prepare_dataset(
            raw_dataset_path,
            sample_size=2_000_000,
            max_lines=3_000_000
        )
    except FileNotFoundError:
        print(f"Error: Input file '{raw_dataset_path}' not found!")
        return
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return

    # Preprocess the text
    print("Cleaning and normalizing text...")
    preprocessed_data = [preprocess_hindi_text(line) for line in tqdm(raw_data)]

    # Save the preprocessed dataset
    with open(preprocessed_path, "w", encoding="utf-8") as file:
        file.write("\n".join(preprocessed_data))

    # Initialize and train our custom BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=5000)
    tokenizer.train(preprocessed_data, min_frequency=2)
    
    # Save the tokenizer
    config_path = output_dir / "hindi_encoder.json"
    tokenizer.save(str(config_path))
    
    # Test the tokenizer
    #test_text = "नमस्ते भारत! यह एक परीक्षण वाक्य है।"
    test_text = "फिर पानी भी कम मात्रा में"
    test_tokenizer(tokenizer, test_text)
    
    return tokenizer

def load_tokenizer(config_path):
    """
    Loads a previously trained tokenizer from a configuration file.
    
    Args:
        config_path (str): Path to the tokenizer configuration file
    
    Returns:
        Tokenizer: Loaded tokenizer
    """
    return BPETokenizer.load(config_path)

if __name__ == "__main__":
    main() 