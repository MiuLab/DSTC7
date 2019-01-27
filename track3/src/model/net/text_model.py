import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing
import random

from ..components.SimpleGRUEncoder import Simple_GRU_Encoder
from ..components.DocReader import DocReader
from ..components.FeatureEncoder import I3D_Flow_Encoder, I3D_RGB_Encoder, VGGish_Encoder, Caption_Encoder
from ..components.QuestionEncoder import Question_Encoder
from ..components.BiattentionGenerator import Biattention_Generator
# TODO: Add trainable word embedding
from ..components.ban import build_ban
from ..components.fusion_attention import WordAttention, FullyAttention, MultiHeadedAttention

class Fusion(nn.Module):
    '''
        Fuse vectors in the same vector space by linear combination
    '''
    # TODO: Attention fusion
    def __init__(self, fusion_candidate_size=6):
        super(Fusion, self).__init__()
        self.weight = nn.Parameter(torch.randn(fusion_candidate_size))
    def forward(self, *inputs):
        output = 0
        # Apply weight
        for i, (input, w) in enumerate(zip(inputs, self.weight)):
            output += w * input
        # Weighted sum
        return output / torch.sum(self.weight)

class MultiChannelFusion(nn.Module):
    def __init__(self, fusion_candidate_size=6):
        super(MultiChannelFusion, self).__init__()
        self.conv1 = nn.Conv1d(fusion_candidate_size, 1, 1)
    def forward(self, *inputs):
        inputs = torch.stack(inputs, dim=1)
        mix = self.conv1(inputs).squeeze(1)
        return mix

class TextModel(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 word_embed_size, 
                 voc_size, 
                 dropout_rate=0.2, 
                 video='conv',
                 gen_attn='linear'):
        super(TextModel, self).__init__()
        i3d_flow_dim, i3d_rgb_dim, vggish_dim = 2048, 2048, 128
        video_dim = i3d_flow_dim + i3d_rgb_dim + vggish_dim
        video_dim_reduced = hidden_size 
        fusion_hidden_size = hidden_size * 2

        self.caption_encoder = DocReader(
                    input_size=word_embed_size*2, hidden_size=hidden_size, 
                    num_layers=1, dropout=0, bidirectional=True)
        self.question_encoder = DocReader(
                    input_size=word_embed_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        self.dialog_encoder = DocReader(
                    input_size=word_embed_size*2, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        
        self.fusion_reader = DocReader(
                    input_size=fusion_hidden_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)

        self.self_reader = DocReader(
                    input_size=fusion_hidden_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        
        self.flow_reader = DocReader(
                    input_size=i3d_flow_dim, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)

        self.rgb_reader = DocReader(
                    input_size=i3d_rgb_dim, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True
        )
        self.vgg_reader = DocReader(
                    input_size=vggish_dim, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True
        )

        self.c_q_attn = WordAttention(word_embed_size)
        self.d_q_attn = WordAttention(word_embed_size)
        self.context_fusion = MultiHeadedAttention(8, fusion_hidden_size)
        self.self_fusion = MultiHeadedAttention(8, fusion_hidden_size)
        #self.fusion = MultiChannelFusion(5)
        self.fusion = MultiChannelFusion(5)

        # answer generator
        self.answer_generator = Biattention_Generator(
                        input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                        voc_size=voc_size, attention=gen_attn)
        # dropouts
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, 
                ground_truth_answer, ground_truth_answer_seq_helper, word_helper=None):
        # dropout
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)

        # attn caption information and encode it
        caption_attn = self.c_q_attn(caption, question)
        caption_all, encoded_caption = self.caption_encoder.forward(torch.cat((caption, caption_attn), dim=2))
        
        # attn dialog information and encode it
        dialog_attn = self.d_q_attn(dialog_history, question)
        dialog_all, encoded_dialog = self.dialog_encoder.forward(torch.cat((dialog_history, dialog_attn), dim=2))
        
        # read question
        question_all, encoded_question = self.question_encoder.forward(question)
        
        # read video 
        #flow_all, encoded_flow = self.flow_reader(i3d_flow)
        #rgb_all, encoded_rgb = self.rgb_reader(i3d_rgb)
        #vgg_all, encoded_vgg = self.vgg_reader(vggish)

        # form context
        context = torch.cat((caption_all, dialog_all), dim=1)

        # context fusion
        context_fusion = self.context_fusion(context, question_all, question_all)
        context_fusion_all, encoded_context_fusion = self.fusion_reader(context_fusion)
        context_fusion_all = context_fusion + context_fusion_all

        # self fusion
        self_fusion = self.self_fusion(context_fusion_all, context_fusion_all, context_fusion_all)
        self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
        self_fusion_all = self_fusion + self_fusion_all

        fusion_all = self_fusion_all
        encoded_fusion = self.fusion(encoded_caption, 
                                     encoded_dialog, 
                                     encoded_question, 
                                     encoded_context_fusion, 
                                     encoded_self_fusion
                                     )
                                     #encoded_flow,
                                     #encoded_rgb,
                                     #encoded_vgg)

        answer_word_probabilities = self.answer_generator.forward(fusion_all, encoded_fusion, 
            ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer)
        #yn_prob = self.yn_pred(F.relu(encoded_fusion))
        return answer_word_probabilities, None

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, word_helper, vocab_weight):
        # dropout
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)

        # attn caption information and encode it
        caption_attn = self.c_q_attn(caption, question)
        caption_all, encoded_caption = self.caption_encoder.forward(torch.cat((caption, caption_attn), dim=2))
        
        # attn dialog information and encode it
        dialog_attn = self.d_q_attn(dialog_history, question)
        dialog_all, encoded_dialog = self.dialog_encoder.forward(torch.cat((dialog_history, dialog_attn), dim=2))
        
        # read question
        question_all, encoded_question = self.question_encoder.forward(question)
        
        # read video 
        flow_all, encoded_flow = self.flow_reader(i3d_flow)
        rgb_all, encoded_rgb = self.rgb_reader(i3d_rgb)
        vgg_all, encoded_vgg = self.vgg_reader(vggish)

        # form context
        context = torch.cat((caption_all, dialog_all), dim=1)

        # context fusion
        context_fusion = self.context_fusion(context, question_all, question_all)
        context_fusion_all, encoded_context_fusion = self.fusion_reader(context_fusion)
        context_fusion_all = context_fusion + context_fusion_all

        # self fusion
        self_fusion = self.self_fusion(context_fusion_all, context_fusion_all, context_fusion_all)
        self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
        self_fusion_all = self_fusion + self_fusion_all

        fusion_all = self_fusion_all
        encoded_fusion = self.fusion(encoded_caption, 
                                     encoded_dialog, 
                                     encoded_question, 
                                     encoded_context_fusion, 
                                     encoded_self_fusion)
                                     #encoded_flow,
                                     #encoded_rgb,
                                     #encoded_vgg)

        answer_indices, _ = self.answer_generator.generate(fusion_all, encoded_fusion, word_helper)
    
        return answer_indices
        