import torch
import torch.nn as nn

# Sample vocabulary size and hidden size
vocab_size = 10    # Assume 10 unique words in our vocab
hidden_size = 5    # The size of the hidden state

'''
    With a hidden size of 5, the hidden state might store:

    Index   Possible Meaning (Not directly interpretable)
    0	    Sentiment (e.g., positive tone)
    1	    Subject (e.g., related to "AI")
    2	    Verb Tense (e.g., present tense)
    3	    Sentence Structure (e.g., short phrase)
    4	    Uncertainty/Confidence level
'''

# Define the encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        '''
            GRU and LSTM are forms of RNN
            Instead of GRU in the embedding layer, we can also use LSTM or RNN as well

            RNN have problem of vanishing/exploding gradient

            LSTM are more computationally expensive but better at remembering and forgetting relationships in even long sequences.
            GRU are faster than LSTM as they have less number of gates and also responds with the same quality of output as LSTM but for shorter sequences only.
        '''
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.embedding(input)  # Convert words to embeddings
        '''
            converts shape of input_sequence from (1, 4) to (1, 4, 5)
            4: size of input sequence
            5: hidden_shape
            It means each word respresentation is converted to a vector of size hidden_shape i.e. 5 here
        '''
        print(f"Embeddings for Input Sequence: \n {embedded}")

        embedded = self.dropout(embedded) # drop some embeddings
        '''
            Dropout randomly sets some values in the tensor to zero with probability p (0.1 here).  
            The remaining values are scaled by 1 / (1 - p) to maintain overall magnitude.  
            This helps prevent overfitting by ensuring the model doesn't rely too much on specific neurons.

        '''
        print(f"Embeddings after dropout: \n {embedded}")
        
        output, hidden = self.gru(embedded)  # Pass through GRU Network
        return output, hidden

# # Create encoder model
# encoder = EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)

# # Example sentence "I love AI <EOS>" -> tokenized as [2, 3, 4, 1]
# input_tensor = torch.tensor([[2, 3, 4, 1]])  # Shape: (batch_size=1, seq_length=4)

# # Forward pass through encoder
# output, hidden = encoder(input_tensor)

# # Print results
# print("Output Tensor (Hidden states at each time step):")
# print(output)  # Shape: (1, 4, 5)

# print("\nFinal Hidden State (last hidden state of GRU):")
# print(hidden)  # Shape: (1, 1, 5)
