import torch
from torch import nn
import torch.multiprocessing as multiprocessing
import random

from ..components.TransformerEncoder import TransformerEncoder
#from ..components.SimpleGRUEncoder import Simple_GRU_Encoder
from ..components.FeatureEncoder import I3D_Flow_Encoder, I3D_RGB_Encoder, VGGish_Encoder, Caption_Encoder
from ..components.QuestionEncoder import Question_Encoder
from ..components.AnswerGenerator import Answer_Generator
# TODO: Add trainable word embedding

def pooled_dim(dim, kernel_size, stride, padding, dilation):
    pooled = dim + 2*padding - dilation * (kernel_size - 1) - 1
    return int(pooled / stride) + 1

class TransformerModel(nn.Module):
    '''
        SimpleModel:
            Map all related video, audio feature into same vector space(e.g. fusion_hidden_size dimension space)
            And combine them together by weighted sum as answer generator's first hidden vector
            And use this hidden vector to generate answer
    '''
    def __init__(self, fusion_hidden_size, word_embed_size, voc_size, attention='fusion'):
        super(TransformerModel, self).__init__()
        i3d_flow_dim, i3d_rgb_dim, vggish_dim = 2048, 2048, 128
        
        kernel_size = 8
        stride=8
        padding=0
        dilation=1
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        pooled_i3d_dim = pooled_dim(i3d_flow_dim, kernel_size, stride, padding, dilation)

        self.i3d_flow_encoder = TransformerEncoder(
                    d_model=pooled_i3d_dim, d_hidden=fusion_hidden_size, N=3, h=8)
        self.i3d_rgb_encoder = TransformerEncoder(
                    d_model=pooled_i3d_dim, d_hidden=fusion_hidden_size, N=3, h=8)
        self.vggish_encoder = TransformerEncoder(
                    d_model=vggish_dim, d_hidden=fusion_hidden_size, N=3, h=4)
        #self.i3d_flow_encoder = I3D_Flow_Encoder(
        #            input_size=i3d_flow_dim, hidden_size=fusion_hidden_size, 
        #            num_layers=1, dropout=0, bidirectional=False)
        #self.i3d_rgb_encoder = I3D_RGB_Encoder(
        #            input_size=i3d_rgb_dim, hidden_size=fusion_hidden_size,
        #            num_layers=1, dropout=0, bidirectional=False)
        #self.vggish_encoder = VGGish_Encoder(
        #            input_size=vggish_dim, hidden_size=fusion_hidden_size,
        #            num_layers=1, dropout=0, bidirectional=False)

        self.caption_encoder = TransformerEncoder(
                    d_model=word_embed_size, d_hidden=fusion_hidden_size, N=3, h=6)
        self.question_encoder = TransformerEncoder(
                    d_model=word_embed_size, d_hidden=fusion_hidden_size, N=3, h=6)
        self.history_encoder = TransformerEncoder(
                    d_model=word_embed_size, d_hidden=fusion_hidden_size, N=3, h=6)
        self.answer_generator = Answer_Generator(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                    voc_size=voc_size, attention=attention)
    
    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, dialog_history_seq_helper, 
                ground_truth_answer, ground_truth_answer_seq_helper):
        i3d_flow = self.pool(i3d_flow)
        i3d_rgb = self.pool(i3d_rgb)
        _, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        _, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        _, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        _, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        _, encoded_question = self.question_encoder.forward(question, question_seq_helper)
        if len(dialog_history.size()) == 3:
            _, dialog_history = self.history_encoder.forward(dialog_history)

        answer_word_probabilities, new_dialog_history = self.answer_generator.forward(encoded_i3d_flow, encoded_i3d_rgb, 
            encoded_vggish, encoded_caption, encoded_question, dialog_history, 
            ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer)

        return answer_word_probabilities, new_dialog_history

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, dialog_history_seq_helper,word_helper):
        # Encode past information
        i3d_flow = self.pool(i3d_flow)
        i3d_rgb = self.pool(i3d_rgb)

        _, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        _, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        _, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        _, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        _, encoded_question = self.question_encoder.forward(question, question_seq_helper)
        if len(dialog_history.size()) == 3:
            _, dialog_history = self.history_encoder.forward(dialog_history)
        # Generate answers
        answer_indices, _ = self.answer_generator.generate(encoded_i3d_flow, encoded_i3d_rgb, 
            encoded_vggish, encoded_caption, encoded_question, dialog_history, word_helper)
        
        return answer_indices
        











    