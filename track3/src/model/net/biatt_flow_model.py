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
from ..components.fusion_attention import WordAttention, FullyAttention

class BiattFlowModel(nn.Module):
    def __init__(self, hidden_size, word_embed_size, voc_size, dropout_rate=0.2):
        super(BiattFlowModel, self).__init__()
        i3d_flow_dim, i3d_rgb_dim, vggish_dim = 2048, 2048, 128
        video_dim = i3d_flow_dim + i3d_rgb_dim + vggish_dim
        video_hidden = hidden_size * 2

        self.video_encoder = DocReader(
                        input_size=video_dim, hidden_size=video_hidden,
                        num_layers=1, dropout=0, bidirectional=True)
        self.caption_encoder = DocReader(
                    input_size=word_embed_size*2, hidden_size=hidden_size, 
                    num_layers=1, dropout=0, bidirectional=True)
        self.question_encoder = DocReader(
                    input_size=word_embed_size*2, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        self.text_encoder = DocReader(
                    input_size=word_embed_size*2 + hidden_size*4, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        
        self.fusion_reader = DocReader(
                    input_size=hidden_size*2, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        self.self_reader = DocReader(
                    input_size=hidden_size*2, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        #self.ban = build_ban(video_dim, fusion_hidden_size, gamma=8)
        self.c_q_attn = WordAttention(word_embed_size)
        self.q_c_attn = WordAttention(word_embed_size)
        self.q_c_fusion = FullyAttention(word_embed_size*2 + hidden_size*2, 256)

        self.image_fusion = FullyAttention(hidden_size * 4, 256)
        #self.self_fusion = FullyAttention(fusion_hidden_size * 2, 256)
        # answer generator
        self.answer_generator = Biattention_Generator(
                        input_size=word_embed_size, hidden_size=hidden_size*2, 
                        voc_size=voc_size)
        # dropouts
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog, 
                ground_truth_answer, ground_truth_answer_seq_helper):
        # dropout
        i3d_flow = self.dropout(i3d_flow)
        i3d_rgb = self.dropout(i3d_rgb)
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog = self.dropout(dialog)

        # cat dialog and question
        caption = torch.cat((caption, dialog), dim=1)
        
        # video reader
        video = torch.cat((i3d_flow, i3d_rgb, vggish), dim=2)
        video_all, encoded_video = self.video_encoder.forward(video)
        video_all = self.dropout(video_all)

        # attn caption information and encode it
        cq_attn = self.c_q_attn(caption, question)
        caption_all, _ = self.caption_encoder.forward(torch.cat((caption, cq_attn), dim=2))
        caption_all = self.dropout(caption_all)

        # attn question information and encode it
        qc_attn = self.q_c_attn(question, caption)
        question_all, _ = self.question_encoder.forward(torch.cat((question, qc_attn), dim=2))
        question_all = self.dropout(question_all)
        
        # attn to dialog, caption
        c_history = torch.cat((caption, cq_attn, caption_all), dim=2)
        q_history = torch.cat((question, qc_attn, question_all), dim=2)
        qc_fusion = self.q_c_fusion(q_history, c_history, caption_all)

        # read question and cat all text information
        text, _ = self.text_encoder.forward(torch.cat((question, qc_attn, question_all, qc_fusion), dim=2))
        text = self.dropout(text)

        # video fusion
        text_history = torch.cat((qc_fusion, text), dim=2)
        video_fusion = self.image_fusion(video_all, text_history, text)
        video_fusion_all, encoded_video_fusion = self.fusion_reader(video_fusion)
        video_fusion_all = self.dropout(video_fusion_all)
        encoded_video_fusion = self.dropout(encoded_video_fusion)

        # self fusion
        '''video_history = torch.cat((video_all, video_fusion_all), dim=2)
        self_fusion = self.self_fusion(video_history, video_history, video_fusion_all)
        self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
        self_fusion_all = self.dropout(self_fusion_all)
        encoded_self_fusion = self.dropout(encoded_self_fusion)
        '''
        # cat hidden state
        #text_prior = encoded_question
        text_prior = None

        fusion_all, encoded_fusion = video_fusion_all, encoded_video_fusion

        answer_word_probabilities, new_dialog_history = self.answer_generator.forward(fusion_all, encoded_video_fusion, 
            ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer,
            text_prior = text_prior)

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
        video_all, encoded_video = self.video_encoder.forward(video)
        video_all = self.dropout(video_all)

        # attn caption information and encode it
        caption_attn = self.c_q_attn(caption, question)
        caption_all, encoded_caption = self.caption_encoder.forward(torch.cat((caption, caption_attn), dim=2))
        
        # attn dialog information and encode it
        dialog_attn = self.d_q_attn(dialog_history, question)
        dialog_all, encoded_dialog = self.dialog_encoder.forward(torch.cat((dialog_history, dialog_attn), dim=2))
        
        # read question and cat all text information
        question_all, encoded_question = self.question_encoder.forward(question)
        text = torch.cat((caption_all, question_all, dialog_all), dim=1)
        text = self.dropout(text)

        # video fusion
        video_fusion = self.image_fusion(video_all, text, text)
        video_fusion_all, encoded_video_fusion = self.fusion_reader(video_fusion)
        video_fusion_all = self.dropout(video_fusion_all)
        encoded_video_fusion = self.dropout(encoded_video_fusion)

        # self fusion
        '''video_history = torch.cat((video_all, video_fusion_all), dim=2)
        self_fusion = self.self_fusion(video_history, video_history, video_fusion_all)
        self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
        self_fusion_all = self.dropout(self_fusion_all)
        encoded_self_fusion = self.dropout(encoded_self_fusion)
        '''
        # cat hidden state
        text_prior = encoded_question

        fusion_all, encoded_fusion = video_fusion_all, encoded_video_fusion

        # Generate answers
        answer_indices, _ = self.answer_generator.generate(fusion_all, encoded_fusion, word_helper)
    
        return answer_indices
        