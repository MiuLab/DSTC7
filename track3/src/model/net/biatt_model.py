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


class BiattModel(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 word_embed_size, 
                 voc_size, 
                 dropout_rate=0.2, 
                 video='conv',
                 gen_attn='linear'):
        super(BiattModel, self).__init__()
        i3d_flow_dim, i3d_rgb_dim, vggish_dim = 2048, 2048, 128
        video_dim = i3d_flow_dim + i3d_rgb_dim + vggish_dim
        video_dim_reduced = hidden_size 
        fusion_hidden_size = hidden_size * 2
        

        # kernels 256 stride (1,16) (11,32)
        self.video = video
        if video == 'conv':
            self.bn = nn.BatchNorm1d(video_dim)
            self.video_encoder = nn.Conv1d(in_channels=video_dim, 
                                        out_channels=fusion_hidden_size,
                                        kernel_size=16,
                                        stride=4,
                                        )
        elif video == 'lstm':
            self.video_encoder = DocReader(
                        input_size=video_dim, hidden_size=hidden_size,
                        num_layers=1, dropout=0, bidirectional=True)
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
        #self.text_reader = DocReader(
        #            input_size=fusion_hidden_size, hidden_size=hidden_size,
        #            num_layers=1, dropout=0, bidirectional=True
        #)
        self.self_reader = DocReader(
                    input_size=fusion_hidden_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        #self.ban = build_ban(video_dim, fusion_hidden_size, gamma=8)
        self.c_q_attn = WordAttention(word_embed_size)
        self.d_q_attn = WordAttention(word_embed_size)
        #self.image_fusion = FullyAttention(fusion_hidden_size, 256)
        #self.self_fusion = FullyAttention(fusion_hidden_size * 2, 256)
        self.image_fusion = MultiHeadedAttention(8, fusion_hidden_size)
        self.self_fusion = MultiHeadedAttention(8, fusion_hidden_size)
        self.fusion = Fusion(6)

        # answer generator
        self.answer_generator = Biattention_Generator(
                        input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                        voc_size=voc_size, attention=gen_attn)
        # dropouts
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, 
                ground_truth_answer, ground_truth_answer_seq_helper):
        # dropout
        i3d_flow = self.dropout(i3d_flow)
        i3d_rgb = self.dropout(i3d_rgb)
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)

        # video reader
        video = torch.cat((i3d_flow, i3d_rgb, vggish), dim=2)
        if self.video == 'conv':
            video = video.permute(0, 2, 1).contiguous()
            video = self.bn(video)
            video_all = self.video_encoder.forward(video)
            video_all = video_all.permute(0, 2, 1).contiguous()
            encoded_video = video_all.mean(1)
        else:
            video_all, encoded_video = self.video_encoder.forward(video)
        video_all = self.dropout(video_all)
        encoded_video = self.dropout(encoded_video)

        # attn caption information and encode it
        caption_attn = self.c_q_attn(caption, question)
        caption_all, encoded_caption = self.caption_encoder.forward(torch.cat((caption, caption_attn), dim=2))
        caption_all = self.dropout(caption_all)
        encoded_caption = self.dropout(encoded_caption)
        
        # attn dialog information and encode it
        dialog_attn = self.d_q_attn(dialog_history, question)
        dialog_all, encoded_dialog = self.dialog_encoder.forward(torch.cat((dialog_history, dialog_attn), dim=2))
        dialog_all = self.dropout(dialog_all)
        encoded_dialog = self.dropout(encoded_dialog)
        
        # read question and cat all text information
        question_all, encoded_question = self.question_encoder.forward(question)
        question_all = self.dropout(question_all)
        encoded_question = self.dropout(encoded_question)
        text = torch.cat((caption_all, question_all, dialog_all), dim=1)

        # video fusion
        video_fusion = self.image_fusion(video_all, text, text)
        video_fusion_all, encoded_video_fusion = self.fusion_reader(video_fusion)
        video_fusion_all = self.dropout(video_fusion_all)
        encoded_video_fusion = self.dropout(encoded_video_fusion)

        # self fusion
        self_fusion = self.self_fusion(video_fusion_all, video_fusion_all, video_fusion_all)
        self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
        self_fusion_all = self.dropout(self_fusion_all)
        encoded_self_fusion = self.dropout(encoded_self_fusion)

        #text_prior = encoded_question

        fusion_all = self_fusion_all
        encoded_fusion = self.fusion(encoded_caption, 
                                     encoded_dialog,  
                                     encoded_question,
                                     encoded_video,
                                     encoded_video_fusion,
                                     encoded_self_fusion
                                     )
        #fusion_all = video_fusion_all
        #encoded_fusion = encoded_video_fusion
        #encoded_fusion = encoded_text_fusion # (v2)

        answer_word_probabilities, new_dialog_history = self.answer_generator.forward(fusion_all, encoded_fusion, 
            ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer)

        return answer_word_probabilities, new_dialog_history

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, word_helper):
        # dropout
        i3d_flow = self.dropout(i3d_flow)
        i3d_rgb = self.dropout(i3d_rgb)
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)

        # video reader
        video = torch.cat((i3d_flow, i3d_rgb, vggish), dim=2)
        if self.video == 'conv':
            video = video.permute(0, 2, 1).contiguous()
            video = self.bn(video)
            video_all = self.video_encoder.forward(video)
            video_all = video_all.permute(0, 2, 1).contiguous()
            encoded_video = video_all.mean(1)
        else:
            video_all, encoded_video = self.video_encoder.forward(video)
        video_all = self.dropout(video_all)
        encoded_video = self.dropout(encoded_video)

        # attn caption information and encode it
        caption_attn = self.c_q_attn(caption, question)
        caption_all, encoded_caption = self.caption_encoder.forward(torch.cat((caption, caption_attn), dim=2))
        caption_all = self.dropout(caption_all)
        encoded_caption = self.dropout(encoded_caption)
        
        # attn dialog information and encode it
        dialog_attn = self.d_q_attn(dialog_history, question)
        dialog_all, encoded_dialog = self.dialog_encoder.forward(torch.cat((dialog_history, dialog_attn), dim=2))
        dialog_all = self.dropout(dialog_all)
        encoded_dialog = self.dropout(encoded_dialog)
        
        # read question and cat all text information
        question_all, encoded_question = self.question_encoder.forward(question)
        question_all = self.dropout(question_all)
        encoded_question = self.dropout(encoded_question)
        text = torch.cat((caption_all, question_all, dialog_all), dim=1)

        # video fusion
        video_fusion = self.image_fusion(video_all, text, text)
        video_fusion_all, encoded_video_fusion = self.fusion_reader(video_fusion)
        video_fusion_all = self.dropout(video_fusion_all)
        encoded_video_fusion = self.dropout(encoded_video_fusion)

        # self fusion
        self_fusion = self.self_fusion(video_fusion_all, video_fusion_all, video_fusion_all)
        self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
        self_fusion_all = self.dropout(self_fusion_all)
        encoded_self_fusion = self.dropout(encoded_self_fusion)

        fusion_all = self_fusion_all
        encoded_fusion = self.fusion(encoded_caption, 
                                     encoded_dialog,  
                                     encoded_question,
                                     encoded_video,
                                     encoded_video_fusion,
                                     encoded_self_fusion
                                     )
        #fusion_all = video_fusion_all
        #encoded_fusion = encoded_video_fusion
        #encoded_fusion = encoded_text_fusion # (v2)
        # Generate answers
        answer_indices, _ = self.answer_generator.generate(fusion_all, encoded_fusion, word_helper)
    
        return answer_indices
        