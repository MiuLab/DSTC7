import torch
import torch.nn as nn

class DocReader(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 bidirectional=True,
                 dropout = 0.6,
                 rnn_type='lstm'):
        super(DocReader, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True,
                              dropout=dropout)
        elif rnn_type =='lstm':
            self.rnn = nn.LSTM(input_size,
                               hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True,
                               dropout=dropout)
        else:
            print('Unexpected rnn type')
            exit()

    def forward(self, input, hidden=None):
        '''
        Args:  
            input: (batch, seq, in_dim)
            hidden: None or (batch, hidden)
        Return:
            output: (batch, seq, hidden)
            hidden: (batch, hidden)
        '''
        batch = input.size(0)
        use_cuda = input.is_cuda
        if hidden is None:
            hidden = self.init_hidden(batch, use_cuda)
        
        output, hidden = self.rnn(input, hidden)
        if self.rnn_type == 'lstm':
            h = hidden[0]
        
        if self.bidirectional:
            h_all = output.contiguous().view(batch, -1, self.hidden_size, 2)
            h_f = h_all[:, -1, :, 0]
            h_b = h_all[:, 0, :, 1]
            h = torch.cat((h_f, h_b), dim=1)
        else:
            h = h.permute(1, 0, 2)
            # h: (batch, layer * direction, hidden)
            h = torch.squeeze(h, dim=1)
        return output, h

    def init_hidden(self, batch_size, use_cuda):
        bidirectional = 2 if self.bidirectional else 1
        h = torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size)

        if self.rnn_type == 'gru':
            return h.cuda() if use_cuda else h
        else:
            c = torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size)
            return (h.cuda(), c.cuda()) if use_cuda else (h, c)

if __name__ == '__main__':
    rnn = Docreader(50, 10, 1)