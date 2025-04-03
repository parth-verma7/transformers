import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 10
SOS_token = 0

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        '''
            encoder_outputs : The output of the encoder
            encoder_hidden : The hidden state of the encoder
        '''
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device="cpu").fill_(SOS_token)
        '''
            considering batch_size = 2
            tensor([
                [0],
                [0]], device='cuda:0')
            )

            BUT as per the encoder created, we have batch_size = 1
            tensor([
                [0]
            ], device='cuda:0'
            )
        '''
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input

                # .topk(1) selects the most probable word by finding the index of the highest value.
                _, topi = decoder_output.topk(1)  # (batch_size, 1)

                # Remove extra dimension and detach from computation graph
                decoder_input = topi.squeeze(-1).detach()  # Shape: (batch_size,)
                # This predicted word becomes the next input to the decoder

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    


# encoder_outputs = torch.tensor([[[-0.2161, -0.2695,  0.3997,  0.4220, -0.2460],
#          [ 0.2845, -0.4040,  0.3219,  0.7833, -0.3974],
#          [-0.0427, -0.4339,  0.4572,  0.8163, -0.1818],
#          [ 0.0184, -0.3584,  0.5348,  0.5897, -0.0511]]])

# encoder_hidden = torch.tensor([[[ 0.0184, -0.3584,  0.5348,  0.5897, -0.0511]]])

# decoder = DecoderRNN(hidden_size=5, output_size=10)

# decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)
# print(decoder_outputs)
# print(decoder_hidden)