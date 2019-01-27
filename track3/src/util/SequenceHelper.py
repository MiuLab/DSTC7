import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SequenceHelper():
    '''
        Batch sequence helper
    '''
    def __init__(self, seq_len, max_length, seq_mask):
        self.seq_mask = seq_mask
        self.seq_len = torch.LongTensor(seq_len)
        self.sorted_seq_len, self.indices = torch.sort(self.seq_len, dim=0, descending=True)
        _, self.restore_indices = torch.sort(self.indices, dim=0, descending=False)

    def __repr__(self):
        return "Sequence mask: {}".format(self.seq_mask) + \
               "Sequence length: {}".format(self.seq_len) + \
               "Sorted sequence length: {}".format(self.sorted_seq_len) + \
               "Indices: {}".format(self.indices) + \
               "Restore indices: {}".format(self.restore_indices)

    def sort(self, batch_seq):
        return torch.index_select(batch_seq, dim=0, index=self.indices)

    def unsort(self, sorted_seq):
        return torch.index_select(sorted_seq, dim=0, index=self.restore_indices)

    def sort_and_pack(self, batch_seq):
        sorted_seq = torch.index_select(batch_seq, dim=0, index=self.indices)
        return pack_padded_sequence(sorted_seq, self.sorted_seq_len, batch_first=True)

    def unpack_and_unsort(self, packed_sequence, max_length):
        sorted_seq, _ = pad_packed_sequence(packed_sequence, batch_first=True, total_length=max_length)
        return torch.index_select(sorted_seq, dim=0, index=self.restore_indices)
    
    def cuda(self):
        self.seq_mask = self.seq_mask.cuda()
        self.seq_len = self.seq_len.cuda()
        self.sorted_seq_len = self.sorted_seq_len.cuda()
        self.indices = self.indices.cuda()
        self.restore_indices = self.restore_indices.cuda()
        return self

    def cpu(self):
        self.seq_mask = self.seq_mask.cpu()
        self.seq_len = self.seq_len.cpu()
        self.sorted_seq_len = self.sorted_seq_len.cpu()
        self.indices = self.indices.cpu()
        self.restore_indices = self.restore_indices.cpu()
        return self

if __name__ == "__main__":
    batch_seq = torch.tensor([[8,5,3], [4,0, 0], [5,6,0]])
    seq_len = [3, 1, 2]
    sh = SequenceHelper(batch_seq, seq_len, 3)
    packed = sh.sort_and_pack(batch_seq)
    batch_seq_ = sh.unpack_and_unsort(packed)
    print(batch_seq_)
