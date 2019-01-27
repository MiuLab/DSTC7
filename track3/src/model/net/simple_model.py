import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing
import random

from ..components.SimpleGRUEncoder import Simple_GRU_Encoder
from ..components.FeatureEncoder import I3D_Flow_Encoder, I3D_RGB_Encoder, VGGish_Encoder, Caption_Encoder
from ..components.QuestionEncoder import Question_Encoder
from ..components.AnswerGenerator import Answer_Generator
from ..components.AttentionGenerator import Attention_Generator
from ..components.DocReader import DocReader
# TODO: Add trainable word embedding

class GatedLinear(nn.Module):
    def __init__(self, input_size, cond_size, out_size):
        super(GatedLinear, self).__init__()
        self.fc = nn.Linear(input_size, out_size)
        self.fc_gate = nn.Linear(cond_size, out_size)

    def forward(self, input, condition):
        '''
        Args:
            Input: (batch, seq, input_size)
            condition: (batch, cond_size)
        '''
        y = self.fc(input)
        g = F.sigmoid(self.fc_gate(condition))
        return y * g.unsqueeze(1)

class TextAwareReader(nn.Module):
    def __init__(self, input_size, cond_size, out_size, hidden_size):
        super(TextAwareReader, self).__init__()
        self.reader = DocReader(input_size=out_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                dropout=0,
                                bidirectional=True)
        self.glu = GatedLinear(input_size=input_size,
                               cond_size=cond_size,
                               out_size=out_size)

    def forward(self, video, text):
        '''
        Args:
            video: (batch, seq, input_size)
            text: (batch, cond_size)
        '''
        video_gated = self.glu(video, text)
        return self.reader(video_gated)

class SimpleModel(nn.Module):
    '''
        SimpleModel:
            Map all related video, audio feature into same vector space(e.g. fusion_hidden_size dimension space)
            And combine them together by weighted sum as answer generator's first hidden vector
            And use this hidden vector to generate answer
    '''
    def __init__(self, fusion_hidden_size, word_embed_size, voc_size, attention='fusion', dropout_rate=0.5, feature_set=0):
        super(SimpleModel, self).__init__()
        i3d_flow_dim, i3d_rgb_dim, vggish_dim = 2048, 2048, 128
        self.i3d_flow_encoder = I3D_Flow_Encoder(
                    input_size=i3d_flow_dim, hidden_size=fusion_hidden_size, 
                    num_layers=1, dropout=0, bidirectional=False)
        self.i3d_rgb_encoder = I3D_RGB_Encoder(
                    input_size=i3d_rgb_dim, hidden_size=fusion_hidden_size,
                    num_layers=1, dropout=0, bidirectional=False)
        self.vggish_encoder = VGGish_Encoder(
                    input_size=vggish_dim, hidden_size=fusion_hidden_size,
                    num_layers=1, dropout=0, bidirectional=False)
        
        hidden_size = fusion_hidden_size // 2
        self.caption_encoder = Caption_Encoder(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                    num_layers=1, dropout=0, bidirectional=False)
        self.question_encoder = Question_Encoder(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size,
                    num_layers=1, dropout=0, bidirectional=False)
        self.history_encoder = Simple_GRU_Encoder(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size,
                    num_layers=1, dropout=0, bidirectional=False
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attention = attention
        if attention == 'top_down':
            self.answer_generator = Attention_Generator(
                                    input_size=word_embed_size, hidden_size=fusion_hidden_size,
                                    feature_size=fusion_hidden_size*2, voc_size=voc_size)
        elif attention == 'top_down_all':
            self.answer_generator = Attention_Generator(
                                    input_size=word_embed_size, hidden_size=fusion_hidden_size,
                                    feature_size=fusion_hidden_size, voc_size=voc_size)
        else:
            self.answer_generator = Answer_Generator(
                        input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                        voc_size=voc_size, attention=attention, feature_set=feature_set)

    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, 
                ground_truth_answer, ground_truth_answer_seq_helper, word_helper=None):
        i3d_flow = self.dropout(i3d_flow)
        i3d_rgb = self.dropout(i3d_rgb)
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)

        i3d_flow_all, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        i3d_rgb_all, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        vggish_all, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        caption_all, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        question_all, encoded_question = self.question_encoder.forward(question, question_seq_helper)
        # if dialog history is given as (batch, seq, embedding) 
        if len(dialog_history.size()) == 3: 
            dialog_history_all, dialog_history = self.history_encoder.forward(dialog_history)
        else:
            # dialog_history is (batch, embedding)
            dialog_history_all = dialog_history.unsqueeze(1)

        if self.attention == 'top_down':
            video_len = min(i3d_flow_all.size(1), i3d_rgb_all.size(1))
            attention_vector = torch.cat((i3d_flow_all[:, :video_len, :], i3d_rgb_all[:, :video_len, :]),
                                          dim=2)
            #attention_vector = torch.cat((i3d_flow_all, i3d_rgb_all, vggish_all), dim=1)
            fusion_vector = torch.stack((encoded_i3d_flow, encoded_i3d_rgb, 
                            encoded_vggish, encoded_caption, 
                            encoded_question, dialog_history),
                            dim=1)
            answer_word_probabilities, new_dialog_history = self.answer_generator.forward(attention_vector,
                                                            ground_truth_seq_helper=ground_truth_answer_seq_helper,
                                                            ground_truth_answer_input=ground_truth_answer,
                                                            fusion_vector=fusion_vector)
        elif self.attention == 'top_down_all':
            feature_vector = torch.cat((i3d_flow_all, 
                                        i3d_rgb_all,
                                        vggish_all,
                                        caption_all,
                                        question_all,
                                        dialog_history_all), dim=1)
            answer_word_probabilities, new_dialog_history = self.answer_generator.forward(feature_vector, 
                                                ground_truth_seq_helper=ground_truth_answer_seq_helper,
                                                ground_truth_answer_input=ground_truth_answer)
        else:
            answer_word_probabilities, fusion_vector = self.answer_generator.forward(encoded_i3d_flow, encoded_i3d_rgb, 
                encoded_vggish, encoded_caption, encoded_question, dialog_history, 
                ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer, word_helper=word_helper)

        return answer_word_probabilities, None

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, word_helper, vocab_weight=None):
        i3d_flow = self.dropout(i3d_flow)
        i3d_rgb = self.dropout(i3d_rgb)
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)
        
        i3d_flow_all, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        i3d_rgb_all, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        vggish_all, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        caption_all, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        question_all, encoded_question = self.question_encoder.forward(question, question_seq_helper)
        # if dialog history is given as (batch, seq, embedding) 
        if len(dialog_history.size()) == 3: 
            dialog_history_all, dialog_history = self.history_encoder.forward(dialog_history)
        else:
            # dialog_history is (batch, embedding)
            dialog_history_all = dialog_history.unsqueeze(1)

        if self.attention == 'top_down':
            #attention_vector = torch.cat((i3d_flow_all, i3d_rgb_all, vggish_all), dim=1)
            video_len = min(i3d_flow_all.size(1), i3d_rgb_all.size(1))
            attention_vector = torch.cat((i3d_flow_all[:, :video_len, :], i3d_rgb_all[:, :video_len, :]),
                                          dim=2)
            fusion_vector = torch.stack((encoded_i3d_flow, encoded_i3d_rgb, 
                            encoded_vggish, encoded_caption, 
                            encoded_question, dialog_history),
                            dim=1)
            answer_indices, _ = self.answer_generator.generate(attention_vector, word_helper, fusion_vector=fusion_vector)
        elif self.attention == 'top_down_all':
            feature_vector = torch.cat((i3d_flow_all, 
                                        i3d_rgb_all,
                                        vggish_all,
                                        caption_all,
                                        question_all,
                                        dialog_history_all),
                                        dim=1)
            answer_indices, _ = self.answer_generator.generate(feature_vector, word_helper)
        else:
            # Generate answers
            answer_indices, fusion_vector = self.answer_generator.generate(encoded_i3d_flow, encoded_i3d_rgb, 
                encoded_vggish, encoded_caption, encoded_question, dialog_history, word_helper, vocab_weight=vocab_weight)

        return answer_indices
        