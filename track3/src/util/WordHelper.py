import math
import numpy as np
from pathlib import Path 
from torch import nn
import torch
import copy 
from  tqdm import tqdm
import pickle

# TODO: python autocorrect
# TODO: Build vocabulary by training data
# TODO: Apply text normalization (such as case folding)
class WordHelper(nn.Module):
    def __init__(self, fname, file_format, cacheDir, timestamp, word_embed_size, requires_grad=False, elmo=0):
        super(WordHelper, self).__init__()
        ######### Args ##########
        self.fname = fname
        self.file_format = file_format
        self.cacheDir = Path(cacheDir)
        self.timestamp = timestamp
        self.word_embed_size = word_embed_size
        self.requires_grad = requires_grad
        ######### Vars ##########
        self.voc_size = None
        self.index2word = [] # index -> word mapping
        self.word2index = {} # word -> index mapping 
        self.word_embedding_matrix = None # nn.Embedding
        # Load word vector
        # TODO: Parallel computing
        # NOTE: Pad token is at index 0, SOS is at index 1, EOS is at index 2, UNK is at index 3
        self.index2word.append("<PAD>")
        self.word2index["<PAD>"] = 0
        self.index2word.append("<SOS>")
        self.word2index["<SOS>"] = 1
        self.index2word.append("<EOS>")
        self.word2index["<EOS>"] = 2
        self.index2word.append("<UNK>")
        self.word2index["<UNK>"] = 3
        # Check if cache exist
        
        if self.check_cache_existence():
            self.load_all()
        else:        
            if file_format == "glove":
                word_embedding_matrix = []
                with fname.open('r') as f:
                    for i, line in tqdm(enumerate(f)):
                        splited = line.split(" ")
                        word = splited[0]
                        word_vector = list(map(lambda x: float(x), splited[1:]))
                        word_embedding_matrix.append(word_vector)
                        self.index2word.append(word)
                        self.word2index[word] = i + 4
                # TODO: cache word_embedding_matrix list 

                #  List -> torch.Tensor
                word_embedding_matrix = torch.FloatTensor(word_embedding_matrix)
                self.voc_size, _word_embed_size = word_embedding_matrix.size()
                assert _word_embed_size == word_embed_size
                self.voc_size += 4 
                self.word_embedding_matrix = nn.Embedding(self.voc_size, self.word_embed_size, padding_idx=self.word2index["<PAD>"])
                self.word_embedding_matrix.weight.data[4:] = word_embedding_matrix
                # Set requires_grad
                self.word_embedding_matrix.weight.requires_grad = requires_grad 

            else:
                raise NotImplementedError
            # save word all
            self.save_all()

        self.elmo = elmo
        if elmo:
            from allennlp.modules.elmo import Elmo, batch_to_ids
            options_file = './AVSD_Jim/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            weight_file = './AVSD_Jim/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            elmo_layers = 1 # used for squad
            self.word_embed_size = 1024
            self.elmo = Elmo(options_file, weight_file, elmo_layers, dropout=0)
            self.weight = nn.Parameter(torch.ones(elmo_layers))
            #stdv = 1. / math.sqrt(self.weight.size(0))
            #self.weight.data.uniform_(-stdv, stdv)
            self.elmo.requires_grad = requires_grad
            self.weight.requires_grad = requires_grad
            
    def __repr__(self):
        return "Voc size: {}\nFile format: {}\n".format(self.voc_size, self.file_format)
    
    def save_word_embedding(self):
        # if requires_grad is False, we don't need to use timestamp
        if self.requires_grad == True: 
            fname = "Embedding_{}_{}_{}".format(self.file_format, self.word_embed_size, self.timestamp)
        else:
            fname = "Embedding_{}_{}".format(self.file_format, self.word_embed_size)
        torch.save(self.word_embedding_matrix, str(self.cacheDir / fname))
    
    def load_word_embedding(self):
        # if requires_grad is False, we don't need to use timestamp
        if self.requires_grad == True: 
            fname = "Embedding_{}_{}_{}".format(self.file_format, self.word_embed_size, self.timestamp)
        # else, because we already load word_embedding_matrix in __init__, we don't need to load again
        else:
            fname = "Embedding_{}_{}".format(self.file_format, self.word_embed_size)
        self.word_embedding_matrix = torch.load(str(self.cacheDir / fname))

    def check_cache_existence(self):
        if self.requires_grad == True:
            embedding_fname = "Embedding_{}_{}_{}".format(self.file_format, self.word_embed_size, self.timestamp)
        else:
            embedding_fname = "Embedding_{}_{}".format(self.file_format, self.word_embed_size)
        misc_fname = "WordHelper_{}_{}".format(self.file_format, self.word_embed_size)
        return (self.cacheDir / embedding_fname).exists() and (self.cacheDir / misc_fname).exists()
    
    def save_all(self):
        """
            Save:
                voc_size
                word_embed_size
                index2word
                word2index
                word_embedding_matrix
        """
        # Save timestamp-irrelevant variables
        fname = "WordHelper_{}_{}".format(self.file_format, self.word_embed_size)
        data = {"voc_size": self.voc_size, 
                "word_embed_size": self.word_embed_size, 
                "index2word": self.index2word, 
                "word2index": self.word2index,
                }
        with (self.cacheDir / fname).open("wb") as f:
            pickle.dump(data, f)
        # save word embedding
        self.save_word_embedding()

    def load_all(self):
        # load timestamp-irrelevant variables
        fname = "WordHelper_{}_{}".format(self.file_format, self.word_embed_size)
        with (self.cacheDir / fname).open('rb') as f:
            data = pickle.load(f)

        self.voc_size = data["voc_size"]
        self.word_embed_size = data["word_embed_size"]
        self.index2word = data["index2word"]
        self.word2index = data["word2index"]
        # load word embedding
        self.load_word_embedding()

    def embed(self, indices, use_cuda=False):
        '''
            Input: (*, )
            Output: (*, embed_size)
        '''
        if isinstance(indices, torch.Tensor):
            if indices.is_cuda:
                # Put it into cpu
                indices = indices.cpu()
        else:
            indices = WordHelper.toLongTensor(indices)

        if not self.elmo:
            embedded = self.word_embedding_matrix.forward(indices)
            if use_cuda:
                embedded = embedded.cuda()
            return embedded
        else:
            batch = indices.size(0)
            tokens = []
            for indice in indices:
                token = []
                for index in indice:
                    word = self.index2word[index]
                    token.append(word)
                tokens.append(token)
        
            character_ids = batch_to_ids(tokens)
            embeddings = self.elmo(character_ids)

            embed = embeddings['elmo_representations']
            output = 0
            for i, (input, w) in enumerate(zip(embed, self.weight)):
                output += w * input
        
            if use_cuda:
                output = output.cuda()
            output.detach_()
            return output
    
    @staticmethod
    def toLongTensor(l):
        return torch.LongTensor(l)

    @staticmethod
    def toNumpy(tensor):
        return tensor.numpy()
    
    @staticmethod
    def getSequenceLength(sequences):
        '''
            Input: 
                A list
            Output:
                A list contains each element's length 
        '''
        return list(map(len, sequences))

    
    def pad(self, sequence, max_length):
        '''
            Input: 
                A list of sequence
                max_length
            Output:
                Pad indices to the given length
        '''
        return sequence + [self.word2index["<PAD>"]] * (max_length - len(sequence))
    # TODO: Add autocorrect
    def tokens2indices(self, token_list):
        indices = []
        # Pad SOS in front of tokens
        indices.append(self.word2index["<SOS>"])
        # Append token's index
        for token in token_list:
            if token in self.word2index:
                indices.append(self.word2index[token])
            else:
                # Pad UNK
                indices.append(self.word2index["<UNK>"])
        
        # Pad EOS in back of
        indices.append(self.word2index["<EOS>"])
        return indices

    def indices2tokens(self, indices):
        tokens = []
        for index in indices:
            word = self.index2word[index]
            if word == "<EOS>":
                break
            if word != "<SOS>" and word != "<PAD>" and word != "<UNK>":
                tokens.append(word)
        return tokens

if __name__ == "__main__":
    pass