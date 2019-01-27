"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932
This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
#from language_model import WordEmbedding, QuestionEmbedding
from .biattention import BiAttention
from .fc import FCNet
from .bc import BCNet
#from counting import Counter

class BanModel(nn.Module):
    def __init__(self, v_att, b_net, q_prj, glimpse):
        super(BanModel, self).__init__()
        self.glimpse = glimpse
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        #self.c_prj = nn.ModuleList(c_prj)
        #self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, q_emb):
        """Forward
        v: [batch, num_objs, obj_dim]
        q_emb: [batch_size, seq_length, word_dim]
        return: logits, not probs
        """

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            #atten, _ = logits[:,g,:,:].max(2)
            #embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            #q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        #logits = self.classifier(q_emb.sum(1))
        vec = q_emb.sum(1)

        return vec, att


def build_ban(v_dim=2048, num_hid=1280, gamma=8):
    v_att = BiAttention(v_dim, num_hid, num_hid, gamma)
    b_net = []
    q_prj = []
    c_prj = []
    objects = 200 # minimum number of boxes
    for i in range(gamma):
        b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        #c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    
    # counter = Counter(objects)
    return BanModel(v_att, b_net, q_prj, gamma)

if __name__ == '__main__':
    video = torch.Tensor(32, 200, 2048)
    question = torch.Tensor(32, 63, 1280)
    model = build_ban()
    vec, att = model(video, question)
    print(vec.size())