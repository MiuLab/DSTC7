import torch
from torch import nn
import torch.multiprocessing as multiprocessing
import random

from ..components.cln import LSTM
from ..components.SimpleGRUEncoder import Simple_GRU_Encoder
from ..components.FeatureEncoder import I3D_Flow_Encoder, I3D_RGB_Encoder, VGGish_Encoder, Caption_Encoder
from ..components.QuestionEncoder import Question_Encoder
from ..components.AnswerGenerator import Answer_Generator
# TODO: Add trainable word embedding


class CLNModel(nn.Module):
    '''
        SimpleModel:
            Map all related video, audio feature into same vector space(e.g. fusion_hidden_size dimension space)
            And combine them together by weighted sum as answer generator's first hidden vector
            And use this hidden vector to generate answer
    '''
    def __init__(self, fusion_hidden_size, word_embed_size, voc_size):
        super(CLNModel, self).__init__()
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
        
        self.caption_encoder = LSTM(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                    num_layers=1, dropout=0, bidirectional=False,
                    batch_first=True, image_size=2*fusion_hidden_size)
        self.question_encoder = LSTM(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size,
                    num_layers=1, dropout=0, bidirectional=False,
                    batch_first=True, image_size=2*fusion_hidden_size)
        self.answer_generator = Answer_Generator(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                    voc_size=voc_size)
    
    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, 
                ground_truth_answer, ground_truth_answer_seq_helper):
        _, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        _, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        _, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        
        image_feature = torch.cat((encoded_i3d_flow, encoded_i3d_rgb), dim=1)
        _, (encoded_caption, _) = self.caption_encoder.forward(caption, None, image_feature)
        encoded_caption = encoded_caption.permute(1, 0, 2).squeeze(1)
        _, (encoded_question, _) = self.question_encoder.forward(question, None, image_feature)
        encoded_question =  encoded_question.permute(1, 0, 2).squeeze(1)

        answer_word_probabilities, new_dialog_history = self.answer_generator.forward(encoded_i3d_flow, encoded_i3d_rgb, 
            encoded_vggish, encoded_caption, encoded_question, dialog_history, 
            ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer)

        return answer_word_probabilities, new_dialog_history

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, word_helper):
        # Encode past information
        _, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        _, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        _, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        
        image_feature = torch.cat((encoded_i3d_flow, encoded_i3d_rgb), dim=1)
        _, (encoded_caption, _) = self.caption_encoder.forward(caption, None, image_feature)
        encoded_caption = encoded_caption.permute(1, 0, 2).squeeze(1)
        _, (encoded_question, _) = self.question_encoder.forward(question, None, image_feature)
        encoded_question =  encoded_question.permute(1, 0, 2).squeeze(1)

        # Generate answers
        answer_indices, _ = self.answer_generator.generate(encoded_i3d_flow, encoded_i3d_rgb, 
            encoded_vggish, encoded_caption, encoded_question, dialog_history, word_helper)
        
        return answer_indices
        











    
