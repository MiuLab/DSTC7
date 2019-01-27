import torch
from torch import nn
class Simple_GRU_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(Simple_GRU_Encoder, self).__init__()
        self.stacked_gru = nn.GRU(input_size=input_size, 
            hidden_size=hidden_size, num_layers=num_layers, 
            dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, seq_helper=None):
        '''
            x: PackedSequence object
            seq_helper: help rnn sort and unsort 
        '''
        if seq_helper is None:
            output, h_n = self.stacked_gru.forward(x)
            h_n = h_n.permute(1, 0, 2)
            h_n = torch.squeeze(h_n, dim=1)
            return output, h_n
        else:
            # Get timestep dimension
            max_length = x.size()[1]
            packed_x = seq_helper.sort_and_pack(x)
            packed_output, h_n = self.stacked_gru.forward(packed_x)
            # output is (batch_size, timesteps, hidden_size)
            output = seq_helper.unpack_and_unsort(packed_output, max_length=max_length)
            # TODO: What if num_layers is 2 or bidirectional?
            #  (1, batch, hidden_size) -> (batch, 1, hidden_size)
            h_n = h_n.permute(1, 0, 2)
            h_n = seq_helper.unsort(h_n)
            h_n = torch.squeeze(h_n, dim=1)
            return output, h_n