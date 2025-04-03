import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import EncoderRNN
from extras.decoder import DecoderRNN


SOS_token = 0
vocab_size = 10
hidden_size = 5
MAX_LENGTH = 10


input_tensor = torch.tensor([[2, 3, 4, 1]])
encoder = EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)
encoder_output, encoder_hidden = encoder(input_tensor)


decoder = DecoderRNN(hidden_size=hidden_size, output_size=vocab_size)
decoder_outputs, decoder_hidden, _ = decoder(encoder_output, encoder_hidden)
print(decoder_hidden)