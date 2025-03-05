import torch
from model import build_transformer
from train import greedy_decode
from config import get_config
from dataset import causal_mask
from tokenizers import Tokenizer

# Load configuration
config = get_config()

# Paths to tokenizers
tokenizer_src_path = "/content/drive/MyDrive/transformers/tokenizers/tokenizer_english.json"
tokenizer_tgt_path = "/content/drive/MyDrive/transformers/tokenizers/tokenizer_hindi.json"
model_weights_path = "/content/drive/MyDrive/transformers/weights/tmodel_09.pt"

# Load tokenizers
tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    config['seq_len'],
    config['seq_len'],
    config['d_model']
).to(device)

# Load trained weights
checkpoint = torch.load(model_weights_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Function to translate user input
def translate_sentence(sentence):
    encoded = tokenizer_src.encode(sentence).ids
    encoder_input = torch.tensor(encoded, dtype=torch.int64).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int()
    
    output_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
    translation = tokenizer_tgt.decode(output_tokens.detach().cpu().numpy())
    
    return translation

# Get user input and translate
user_input = "Thank you so much"
translated_sentence = translate_sentence(user_input)
print("Translated to Hindi:", translated_sentence)
