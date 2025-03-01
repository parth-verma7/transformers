import torch, math
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size : int, d_model : int):

        '''
            vocab_size : int : size of the vocabulary
            d_model : int : dimension of the vector model
        '''

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        '''
            multiplying the embeddings by math.sqrt(self.d_model) helps to stabilize the variance of the embeddings and 
            ensures consistency with the scaling used in the attention mechanism, leading to more stable and effective training of the Transformer model.
        '''
        
        '''
            Example:
                vocab_size = 10 (tokens 0-9)
                d_model = 4 (embedding dimension)

            Input example:
                x = tensor([1, 3, 5])
                shape: torch.Size([3])

            Output example:
                return = tensor(
                                [[ 0.8640,  1.2931, -0.7841,  0.9123],
                                [-1.1234,  0.7456,  1.3421, -0.5678],
                                [ 0.3421, -0.8976,  1.1123,  0.4567]]
                            )
                shape: torch.Size([3, 4])
                
            Note: Values are scaled by sqrt(d_model) = sqrt(4) = 2
        '''

        return self.embeddings(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # for every word in seq_len there is a vector of size d_model
        # agar upar dekho toh jiss shape mein embeddings bnn rhi thi vesa hee h
        '''
            torch.Size([3, 5])
            tensor([[0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        '''
        
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float)
        '''
            if seq_len = 3
            position = [0., 1., 2.]
        '''
        position = position.unsqueeze(1)
        '''
            position = [[0.],
                        [1.],
                        [2.]]
        '''

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        '''
            returns a list for ex# tensor([1.0000, 0.0100, 0.0001, 0.0000])
        '''

        # apply sin to even position
        pe[:, 0::2] = torch.sin(position*div_term)
        # apply cos to odd position
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  ## pe are fixed and they should not be learnt during the training process

        '''
            The explicit :x.shape[1] slice is actually crucial! Here's why:

            The positional encodings (self.pe) are pre-computed for the maximum sequence length (e.g., 100 positions),  
            i.e. we can say that once pe are initialised they are initialised with maximum sequence length 
            but your input sequences during training/inference might be shorter (e.g., 50 tokens).

            Without the :x.shape[1] slice:

            If you try to add pe of length 100 to input of length 50, you'd get a shape mismatch error
            The tensors must have matching dimensions for addition
        '''
        return self.dropout(x)


class LayerNormalization(nn.Module):
    '''
        1. Layer normalization helps to stabilize the training process by normalizing the inputs to each layer. 
        This ensures that the inputs have a mean of zero and a standard deviation of one, which can prevent 
        the gradients from exploding or vanishing during backpropagation. 

        2. By normalizing the inputs to each layer, layer normalization can lead to faster convergence during training. 
        This is because the network can learn more efficiently when the inputs to each layer are normalized, 
        reducing the risk of getting stuck in local minima.

        3. Unlike batch normalization, which normalizes across the batch dimension, layer normalization normalizes 
        across the feature dimension. This makes it more suitable for models where the batch size may vary or be small, 
        as it does not rely on batch statistics.
    '''
    def __init__(self, eps: float = 10** -6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)  # for the last layer i.e. after batch
        std = x.std(dim = -1, keepdim = True)
        '''
            x = torch.tensor([
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]
                ],

                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0]
                ]
            ])  
            i.e. (2, 3, 4)

            For the first sub-tensor:
                Mean of [1.0, 2.0, 3.0, 4.0] is (1.0 + 2.0 + 3.0 + 4.0) / 4 = 2.5
                Mean of [5.0, 6.0, 7.0, 8.0] is (5.0 + 6.0 + 7.0 + 8.0) / 4 = 6.5
                Mean of [9.0, 10.0, 11.0, 12.0] is (9.0 + 10.0 + 11.0 + 12.0) / 4 = 10.5

            For the second sub-tensor:
                Mean of [13.0, 14.0, 15.0, 16.0] is (13.0 + 14.0 + 15.0 + 16.0) / 4 = 14.5
                Mean of [17.0, 18.0, 19.0, 20.0] is (17.0 + 18.0 + 19.0 + 20.0) / 4 = 18.5
                Mean of [21.0, 22.0, 23.0, 24.0] is (21.0 + 22.0 + 23.0 + 24.0) / 4 = 22.5

            tensor([
                    [
                        [ 2.5],
                        [ 6.5],
                        [10.5]
                    ],

                    [
                        [14.5],
                        [18.5],
                        [22.5]
                    ]
                ])

            i.e. (2, 3, 1)
        '''
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForward(nn.Module):
    '''
        Purpose of the FeedForward Layer
            Non-Linearity:
            The ReLU activation function introduces non-linearity into the model, allowing it to learn more complex functions and representations.
            
            Dimensionality Transformation:
            The feed-forward layer transforms the input from the model dimension (d_model) to a higher dimension (d_ff) and then back to the model dimension. 
            This allows the model to capture more complex patterns and interactions.
            
            Regularization:
            Dropout is used to prevent overfitting by randomly setting a fraction of the input units to zero during training. 
            This encourages the model to learn more robust features.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x)))) # relu activation function is used
    

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: False):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1)
        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], -1, self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadSelfAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # (Batch, seq_len, d_k) -> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadSelfAttentionBlock, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward)
        '''
            In summary, the output of the self-attention block is indeed passed to the feed-forward block within each EncoderBlock
        '''
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:  # here each layer is an instance of the EncoderBlock
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    '''
        In the multi-head attention block of decoder, value and key coems from encoder whereas query comes from Masked multi head attention block i.e. why it is called cross attention
        whereas in masked multi head attention, query, key and value are the same only i.e. the same as input
    '''
    def __init__(self, self_attention_block: MultiHeadSelfAttentionBlock, cross_attention_block: MultiHeadSelfAttentionBlock, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    '''
        Mentioned as Linear layer in Attention Research Paper
    '''
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    

class Transformer(nn.Module):
    def __init__(
            self, 
            encoder: Encoder, 
            decoder: Decoder, 
            src_embed: InputEmbedding, 
            tgt_embed: InputEmbedding, 
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer
        ):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos((tgt))
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, d_ff: int = 2048, N: int = 6, h: int = 8, dropout: float = 0.1):
    # create the embeddings layer
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    # create the positional encoding layer
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder and decoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadSelfAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout))

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadSelfAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadSelfAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout))
    
    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialise the parameters
    '''
        This method sets the values of the weights to be uniformly distributed within a specific range, 
        which is determined based on the number of input and output units in the weight tensor.
    '''
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

