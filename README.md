# Hindi BPE Tokenizer

This Python script is designed for the preprocessing of Hindi text and the training of a Byte Pair Encoding (BPE) tokenizer specifically tailored for the Hindi language. It automatically fetches and processes a segment of the IndicCorp Hindi dataset.

## Key Features

- **Intelligent Dataset Management**:
  - Downloads the initial 10GB of the IndicCorp Hindi dataset
  - Capable of resuming interrupted downloads
  - Samples 2 million lines from the first 3 million available
  - Includes progress indicators for both downloading and processing

- **Text Preprocessing**:
  - Filters to retain only Hindi characters (Unicode range: \u0900-\u097F)
  - Eliminates digits (both English and Devanagari)
  - Normalizes punctuation (converts Hindi full stops '।' to '.')
  - Cleans up whitespace
  
- **BPE Tokenizer Training**:
  - Enhanced training using numpy's vectorized operations
  - Processes data in batches for improved efficiency
  - Configurable vocabulary size: 5000 tokens
  - Special tokens included: `<pad>`, `<unk>`, `<s>`, `</s>`
  - Minimum token frequency set to 2
  - Tracks progress with compression ratios

## Prerequisites

To install the necessary packages, run:
```
pip install numpy requests tqdm matplotlib
```

## Getting Started

1. Execute the tokenizer training script:
```
python hindi_tokenizer.py
```

2. Utilize the interactive encoder/decoder:
```
python use_tokenizer.py
```

## Directory Layout
```
.
├── hindi_tokenizer.py # Primary training script
├── use_tokenizer.py # Tool for interactive encoding/decoding
├── raw_hindi_dataset.txt # Downloaded dataset (10GB)
└── output/
    ├── preprocessed_hindi.txt # Cleaned text output
    └── hindi_encoder.json # Configuration for the tokenizer
```

## Dataset Information

- **Source**: IndicCorp Hindi Collection
- **URL**: https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt
- **Download Size**: First 10GB of a ~20GB file
- **Training Sample**: 2,000,000 lines from the initial 3 million lines

## Example Usage

### Training the Tokenizer
```
from hindi_tokenizer import main
# Train and retrieve the tokenizer
tokenizer = main()
```

### Utilizing the Trained Tokenizer
```
from hindi_tokenizer import load_tokenizer, encode_text, decode_text
# Load the pre-existing tokenizer
tokenizer = load_tokenizer("output/hindi_encoder.json")
# Encode a sample text
text = "नमस्ते भारत!"
token_ids, tokens = encode_text(tokenizer, text)
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
# Decode back to the original text
decoded_text = decode_text(tokenizer, token_ids)
print(f"Decoded: {decoded_text}")
```

## Technical Insights

### Preprocessing Steps
1. Character filtering: `[^\u0900-\u097F\s।,.!?\-]`
2. Removal of digits: `[0-9०-९]`
3. Normalization of punctuation: `।` → `.`
4. Whitespace normalization

### Tokenizer Settings
- Model Type: Byte Pair Encoding (BPE)
- Vocabulary Size: 5000
- Number of Special Tokens: 4
- Batch Size for Training: 1,000
- Interval for Statistics Tracking: 500
- Utilizes numpy for vectorized operations

### Performance Enhancements
- Vectorized operations based on Numpy
- Batch processing for merge operations
- Optimized memory usage
- Sliding window technique for pair counting
- Pre-allocated arrays for enhanced speed
- Updates to statistics in batches

## Error Management

The script incorporates thorough error handling for:
- Network-related issues during downloads
- Resuming partial downloads
- File input/output operations
- Processing of the dataset
- Verification of compression ratios

## BPE Tokenizer Training Logs
```
(temporary) ➜  erav3-s11-hindi-tokenizer git:(master) ✗ python hindi_tokenizer.py
Sufficient dataset already exists, skipping download.
Step 2: Preprocessing dataset...
Reading and preparing dataset...
Reading lines: 2000005it [00:01, 1093427.18it/s]
Cleaning and normalizing text...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000000/2000000 [00:17<00:00, 114213.87it/s]
Initializing vocabulary...
Computing initial frequencies...
Training BPE:  10%|███████████████████▌                                                                                                                                                                           | 500/4887 [05:05<14:23,  5.08it/s]
Iteration 613
Created token: 'रं' (merged 77,383 times)
Current vocabulary size: 613
Current data size: 266,508,022
Current compression ratio: 1.68
--------------------------------------------------------------------------------
Training BPE:  20%|██████████████████████████████████████▉                                                                                                                                                       | 1000/4887 [06:42<12:09,  5.33it/s]
Iteration 1,113
Created token: 'ह,' (merged 14,825 times)
Current vocabulary size: 1,113
Current data size: 266,508,022
Current compression ratio: 1.74
--------------------------------------------------------------------------------
Training BPE:  31%|██████████████████████████████████████████████████████████▎                                                                                                                                   | 1500/4887 [09:55<06:43,  8.40it/s]
Iteration 1,613
Created token: 'ो ह' (merged 45,509 times)
Current vocabulary size: 1,613
Current data size: 266,508,022
Current compression ratio: 2.24
--------------------------------------------------------------------------------
Training BPE:  41%|█████████████████████████████████████████████████████████████████████████████▊                                                                                                                | 2000/4887 [10:51<05:14,  9.18it/s]
Iteration 2,113
Created token: 'पर्' (merged 26,421 times)
Current vocabulary size: 2,113
Current data size: 266,508,022
Current compression ratio: 2.39
--------------------------------------------------------------------------------
Training BPE:  51%|█████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                            | 2499/4887 [13:17<03:45, 10.61it/s]
Iteration 2,613
Created token: 'हार ' (merged 15,505 times)
Current vocabulary size: 2,613
Current data size: 266,508,022
Current compression ratio: 2.66
--------------------------------------------------------------------------------
Training BPE:  61%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                         | 2999/4887 [14:02<02:48, 11.22it/s]
Iteration 3,113
Created token: 'िले ' (merged 11,115 times)
Current vocabulary size: 3,113
Current data size: 266,508,022
Current compression ratio: 2.79
--------------------------------------------------------------------------------
Training BPE:  72%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                      | 3500/4887 [16:13<01:57, 11.83it/s]
Iteration 3,613
Created token: 'ठाक' (merged 7,706 times)
Current vocabulary size: 3,613
Current data size: 266,508,022
Current compression ratio: 2.93
--------------------------------------------------------------------------------
Training BPE:  82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                  | 4000/4887 [16:54<01:11, 12.48it/s]
Iteration 4,113
Created token: 'ंगठ' (merged 6,185 times)
Current vocabulary size: 4,113
Current data size: 266,508,022
Current compression ratio: 3.03
--------------------------------------------------------------------------------
Training BPE:  92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉               | 4499/4887 [18:52<00:30, 12.78it/s]
Iteration 4,613
Created token: 'बेहद' (merged 4,949 times)
Current vocabulary size: 4,613
Current data size: 266,508,022
Current compression ratio: 3.13
--------------------------------------------------------------------------------
Training BPE: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4887/4887 [19:21<00:00,  4.21it/s]

Training completed. Final vocabulary size: 5000
Final compression ratio: 3.22

Tokenizer Test:
--------------------------------------------------
Original Text: फिर पानी भी कम मात्रा में

Tokens: ['फिर', 'पा', 'नी', 'भी', 'कम', 'मा', 'त्र', 'ा', 'में']
Token IDs: [4947, 215, 225, 210, 450, 172, 1314, 70, 1163]

Decoded Text: फिर पा नी भी कम मा त्र ा में
(temporary) ➜  erav3-s11-hindi-tokenizer git:(master) ✗
```

## BPE Tokenizer Sample Usage Logs
```
(temporary) ➜  erav3-s11-hindi-tokenizer git:(master) ✗ python use_tokenizer.py
Loaded vocabulary size: 5000
Max token ID: 4999
Sample tokens: [(0, '<pad>'), (1, '<unk>'), (2, '<s>'), (3, '</s>'), (4, ' ')]
Hindi Text Encoder/Decoder (type 'quit' to exit)
--------------------------------------------------

Enter Hindi text to encode/decode: शब्दकोश एक बड़ी सूची या किताब होती है

Encoding:
Tokens: ['शब्द', 'को', 'श', 'एक', 'बड़', 'ी', 'सूच', 'ी', 'या', 'कि', 'ताब', 'होत', 'ी', 'है']
Token IDs: [3645, 150, 63, 259, 1767, 72, 3922, 72, 134, 151, 2092, 1484, 72, 132]

Decoding:
Text: शब्द को श एक बड़ ी सूच ी या कि ताब होत ी है

Enter Hindi text to encode/decode: quit
(temporary) ➜  erav3-s11-hindi-tokenizer git:(master) ✗ 
```

## Contributions

We welcome you to report issues or submit pull requests for enhancements.

## License
MIT License
