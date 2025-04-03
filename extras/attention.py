import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size//heads
        
        assert((self.heads_dim)*(self.heads) == self.embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        '''
            Here's how the layers work:
            Input Tensor to nn.Linear: The input to self.values is a tensor of shape (batch_size, seq_len, heads_dim)—that is, 
            the values are projected from the original embedding space to the new heads_dim space (which is 256 in this case).
            Output Tensor from nn.Linear: The output of the linear layer is also of shape (batch_size, seq_len, heads_dim), where heads_dim is 256.
        '''
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.fully_connected_out = nn.Linear(self.embed_size, self.embed_size)

        '''
        If we take embed_size=512 and heads=2
        then our query, keys and values will be ~ (batch_size, seq_length, 256)
        '''

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = query.reshape(N, query_len, self.heads, self.heads_dim)
