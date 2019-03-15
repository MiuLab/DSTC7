import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing
import random

from ..components.SimpleGRUEncoder import Simple_GRU_Encoder
from ..components.TransformerEncoder import TransformerEncoder
from ..components.DocReader import DocReader
from ..components.FeatureEncoder import I3D_Flow_Encoder, I3D_RGB_Encoder, VGGish_Encoder, Caption_Encoder
from ..components.QuestionEncoder import Question_Encoder
from ..components.BiattentionGenerator import Biattention_Generator
from ..components.AttentionGenerator import Attention_Generator
# TODO: Add trainable word embedding
from ..components.fusion_attention import WordAttention, FullyAttention, MultiHeadedAttention

class CatFusion(nn.Module):
    def __init__(self, hidden_size=256, fusion_candidate_size = 6):
        super(CatFusion, self).__init__()
        self.linear = nn.Linear(hidden_size * fusion_candidate_size, hidden_size)
    
    def forward(self, inputs):
        input = torch.cat(inputs, dim=1)
        return self.linear(input)
    
class ProductFusion(nn.Module):
    def __init__(self):
        super(ProductFusion, self).__init__()
    def forward(self, inputs):
        output = inputs[0]
        for input in inputs[1:]:
            output = output * input
        return output

class AddFusion(nn.Module):
    def __init__(self):
        super(AddFusion, self).__init__()
    def forward(self, inputs):
        output = 0
        for input in inputs:
            output += input
        return output

class Fusion(nn.Module):
    '''
        Fuse vectors in the same vector space by linear combination
    '''
    # TODO: Attention fusion
    def __init__(self, fusion_candidate_size=6):
        super(Fusion, self).__init__()
        self.weight = nn.Parameter(torch.randn(fusion_candidate_size))
    def forward(self, inputs):
        output = 0
        # Apply weight
        for i, (input, w) in enumerate(zip(inputs, self.weight)):
            output += w * input
        # Weighted sum
        return output / torch.sum(self.weight)

class Conv11Fusion(nn.Module):
    def __init__(self, fusion_candidate_size=6, channels=6):
        super(Conv11Fusion, self).__init__()
        self.conv1 = nn.Conv1d(fusion_candidate_size, channels, 1)
        self.conv2 = Fusion(channels)
    
    def forward(self, inputs):
        output = 0
        inputs = torch.stack(inputs, dim=1)
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs.permute(1, 0, 2))
        return inputs.squeeze(1)

class Film(nn.Module):
    def __init__(self, input_size, condition_size):
        super(Film, self).__init__()
        self.gamma = nn.Linear(condition_size, input_size)
        self.beta = nn.Linear(condition_size, input_size)
    
    def forward(self, input, condition, seq=False):
        '''
        Args:
            input: (batch, input_size) or (batch, seq, input_size)
            condition: (batch, condition_size)
        '''
        gamma = F.relu(self.gamma(condition))
        beta = F.relu(self.beta(condition))
        gamma = gamma + 1 # shift

        if seq:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return input * gamma + beta

class FusionModel(nn.Module):
    def __init__(self, 
                 fusion_hidden_size, 
                 word_embed_size, 
                 voc_size, 
                 dropout_rate=0.5,
                 encoder = 'gru',
                 pure_text=False,
                 use_i3d=False,
                 word_attn=False,
                 text_fusion=True, 
                 video_fusion=True,
                 cat_fusion=False,
                 fuse_attn=False,
                 fuse='fusion',
                 attn_on_video=False,
                 attn_on_text=False,
                 attn_on_dialog=False,
                 attn_on_fusion=False,
                 attn_on_video_text=False,
                 attn_on_text_dialog=False,
                 attn_on_video_dialog=False,
                 gen_attn='linear'):
        super(FusionModel, self).__init__()
        i3d_flow_dim, i3d_rgb_dim, vggish_dim = 2048, 2048, 128
        hidden_size = fusion_hidden_size // 2

        self.pure_text = pure_text
        self.use_i3d = use_i3d
        self.word_attn = word_attn
        self.text_fusion = text_fusion
        self.video_fusion = video_fusion
        self.cat_fusion = cat_fusion
        self.fuse_attn = fuse_attn
        self.gen_attn = gen_attn
        self.fuse = fuse
        self.attn_on_video = attn_on_video
        self.attn_on_text = attn_on_text
        self.attn_on_dialog = attn_on_dialog
        self.attn_on_fusion = attn_on_fusion
        self.attn_on_video_text = attn_on_video_text
        self.attn_on_text_dialog = attn_on_text_dialog
        self.attn_on_video_dialog = attn_on_video_dialog
        # basic feature reader
        
        if self.word_attn:
            caption_size = word_embed_size*2
            dialog_size = word_embed_size*2
        else:
            caption_size = word_embed_size
            dialog_size = word_embed_size
        
        if encoder == 'transformer':
            self.vggish_encoder = TransformerEncoder(d_model=vggish_dim, d_hidden=fusion_hidden_size, N=3, h=8)
            self.caption_encoder = TransformerEncoder(d_model=caption_size, d_hidden=fusion_hidden_size, N=3, h=6)
            self.dialog_encoder = TransformerEncoder(d_model=dialog_size, d_hidden=fusion_hidden_size, N=3, h=6)
            self.question_encoder = TransformerEncoder(d_model=word_embed_size, d_hidden=fusion_hidden_size, N=3, h=6)
            
            self.fusion_reader = TransformerEncoder(d_model=fusion_hidden_size, d_hidden=fusion_hidden_size, N=3, h=8)
            self.self_reader = TransformerEncoder(d_model=fusion_hidden_size, d_hidden=fusion_hidden_size, N=3, h=8)
            self.vgg_fusion_reader = TransformerEncoder(d_model=fusion_hidden_size, d_hidden=fusion_hidden_size, N=3, h=8)
        else:
            self.vggish_encoder = VGGish_Encoder(
                        input_size=vggish_dim, hidden_size=fusion_hidden_size,
                        num_layers=1, dropout=0, bidirectional=False)
            if self.use_i3d:
                self.i3d_flow_encoder = I3D_Flow_Encoder(
                    input_size=i3d_flow_dim, hidden_size=fusion_hidden_size, 
                    num_layers=1, dropout=0, bidirectional=False)
                self.i3d_rgb_encoder = I3D_RGB_Encoder(
                    input_size=i3d_rgb_dim, hidden_size=fusion_hidden_size,
                    num_layers=1, dropout=0, bidirectional=False)

            self.caption_encoder = Caption_Encoder(
                        input_size=caption_size, hidden_size=fusion_hidden_size, 
                        num_layers=1, dropout=0, bidirectional=False)
            self.dialog_encoder = Simple_GRU_Encoder(
                        input_size=dialog_size, hidden_size=fusion_hidden_size,
                        num_layers=1, dropout=0, bidirectional=False
            )
            self.question_encoder = Question_Encoder(
                        input_size=word_embed_size, hidden_size=fusion_hidden_size,
                        num_layers=1, dropout=0, bidirectional=False)
        
            # attention feature reader
            if self.cat_fusion:
                input_size = fusion_hidden_size * 2
            else:
                input_size = fusion_hidden_size
            self.fusion_reader = DocReader(
                        input_size=input_size, hidden_size=hidden_size,
                        num_layers=1, dropout=0, bidirectional=True, rnn_type='gru')
            self.self_reader = DocReader(
                        input_size=input_size, hidden_size=hidden_size,
                        num_layers=1, dropout=0, bidirectional=True, rnn_type='gru')
            self.vgg_fusion_reader = DocReader(
                    input_size=input_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True, rnn_type='gru')

        if self.pure_text:
            fusion_candidates = 3
        elif self.use_i3d:
            fusion_candidates = 6
        else:
            fusion_candidates = 4

        # attention modules
        if self.word_attn:
            self.c_q_attn = WordAttention(word_embed_size)
            self.d_q_attn = WordAttention(word_embed_size)
        
        if self.text_fusion:
            self.context_fusion = MultiHeadedAttention(8, fusion_hidden_size)
            self.self_fusion = MultiHeadedAttention(8, fusion_hidden_size)
            fusion_candidates += 2
        
        if self.video_fusion:
            self.vgg_fusion = MultiHeadedAttention(8, fusion_hidden_size)
            fusion_candidates += 1
        
        if fuse == 'fusion':
            self.fusion = Fusion(fusion_candidates)
        elif fuse == 'conv11':
            self.fusion = Conv11Fusion(fusion_candidates, fusion_candidates)
        elif fuse == 'conv11_large':
            self.fusion = Conv11Fusion(fusion_candidates, fusion_candidates*3)
        elif fuse == 'cat':
            self.fusion = CatFusion(fusion_hidden_size, fusion_candidates)
        elif fuse == 'add':
            self.fusion = AddFusion()
        elif fuse == 'product':
            self.fusion = ProductFusion()
        else:
            raise NotImplementedError

        # answer generator
        if self.gen_attn == 'topdown':
            self.answer_generator = Attention_Generator(
                                    input_size=word_embed_size, hidden_size=fusion_hidden_size,
                                    feature_size=fusion_hidden_size, voc_size=voc_size)
        else:
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
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)
        
        # read vggish
        vggish_all, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)

        if self.use_i3d:
            flow_all, encoded_flow = self.i3d_flow_encoder(i3d_flow, i3d_flow_seq_helper)
            rgb_all, encoded_rgb = self.i3d_rgb_encoder(i3d_rgb, i3d_rgb_seq_helper)
        
        # attn caption information and encode it
        if self.word_attn:
            caption_attn = self.c_q_attn(caption, question)
            caption = torch.cat((caption, caption_attn), dim=2)
        caption_all, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        # attn dialog information and encode it
        if self.word_attn:
            dialog_attn = self.d_q_attn(dialog_history, question)
            dialog_history = torch.cat((dialog_history, dialog_attn), dim=2)
        dialog_all, encoded_dialog = self.dialog_encoder.forward(dialog_history)
        
        # read question
        question_all, encoded_question = self.question_encoder.forward(question, question_seq_helper)

        # form context
        if self.pure_text:
            fusion_set = [encoded_caption, encoded_dialog, encoded_question]
        elif self.use_i3d:
            fusion_set = [encoded_caption, encoded_dialog, encoded_question, encoded_vggish, encoded_flow, encoded_rgb]
        else:
            fusion_set = [encoded_caption, encoded_dialog, encoded_question, encoded_vggish]

        if self.text_fusion:
            context = torch.cat((caption_all, dialog_all), dim=1)

            # context fusion
            context_fusion = self.context_fusion(context, question_all, question_all)
            if self.cat_fusion:
                context_fusion_all, encoded_context_fusion = self.fusion_reader(torch.cat((context, context_fusion), dim=2))
            else:
                context_fusion_all, encoded_context_fusion = self.fusion_reader(context_fusion)
            context_fusion_all = context_fusion + context_fusion_all

            # self fusion
            self_fusion = self.self_fusion(context_fusion_all, context_fusion_all, context_fusion_all)
            if self.cat_fusion:
                self_fusion_all, encoded_self_fusion = self.self_reader(torch.cat((context_fusion_all, self_fusion), dim=2))
            else:
                self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
            self_fusion_all = self_fusion + self_fusion_all

            fusion_set.append(encoded_context_fusion)
            fusion_set.append(encoded_self_fusion)
            # vgg fusion
            if self.video_fusion:
                vgg_fusion = self.vgg_fusion(vggish_all, self_fusion_all, self_fusion_all)
                if self.cat_fusion:
                    vgg_fusion_all, encoded_vgg_fusion = self.vgg_fusion_reader(torch.cat((vggish_all, vgg_fusion), dim=2))
                else:
                    vgg_fusion_all, encoded_vgg_fusion = self.vgg_fusion_reader(vgg_fusion)
                vgg_fusion_all = vgg_fusion + vgg_fusion_all
        
                fusion_all = vgg_fusion_all
                fusion_set.append(encoded_vgg_fusion)
            else:
                fusion_all = self_fusion_all
        else:
            fusion_all = vggish_all

        if self.fuse_attn:
            encoded_fusion = self.fusion(fusion_set)
        elif self.pure_text:
            encoded_fusion = self.fusion([encoded_caption, encoded_dialog, encoded_question])
        elif self.use_i3d:
            encoded_fusion = self.fusion([encoded_caption, encoded_dialog, encoded_question, encoded_vggish, encoded_flow, encoded_rgb])
        else:
            encoded_fusion = self.fusion([encoded_caption, encoded_dialog, encoded_question, encoded_vggish])

        if self.pure_text:
            features = [caption_all, dialog_all, question_all]
        elif self.use_i3d:
            features = [caption_all, dialog_all, question_all, vggish_all, rgb_all, flow_all]
        else:
            features = [caption_all, dialog_all, question_all, vggish_all]
        '''if self.text_fusion:
            features.append(context_fusion_all)
            features.append(self_fusion_all)
            if self.video_fusion:
                features.append(vgg_fusion_all)
        '''

        feature_vector = torch.cat(features, dim=1)

        if self.gen_attn == 'topdown':
            answer_word_probabilities, new_dialog_history = self.answer_generator.forward(feature_vector, 
                                                ground_truth_seq_helper=ground_truth_answer_seq_helper,
                                                ground_truth_answer_input=ground_truth_answer,
                                                fusion_vector=encoded_fusion)
        else:
            if self.text_fusion and self.attn_on_fusion:
                fusion_all = fusion_all
            elif self.attn_on_video:
                fusion_all = vggish_all
            elif self.attn_on_text:
                fusion_all = caption_all
            elif self.attn_on_dialog:
                fusion_all = dialog_all
            elif self.attn_on_video_text:
                fusion_all = torch.cat([vggish_all, caption_all, question_all], dim=1)
            elif self.attn_on_text_dialog:
                fusion_all = torch.cat([caption_all, dialog_all, question_all], dim=1)
            elif self.attn_on_video_dialog:
                fusion_all = torch.cat([vggish_all, dialog_all, question_all], dim=1)
            else:
                fusion_all = feature_vector
            answer_word_probabilities = self.answer_generator.forward(fusion_all, encoded_fusion, 
                ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer)
        return answer_word_probabilities, None

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, word_helper, vocab_weight):
        # dropout
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog_history = self.dropout(dialog_history)
        
        # read vggish
        vggish_all, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)

        if self.use_i3d:
            flow_all, encoded_flow = self.i3d_flow_encoder(i3d_flow, i3d_flow_seq_helper)
            rgb_all, encoded_rgb = self.i3d_rgb_encoder(i3d_rgb, i3d_rgb_seq_helper)
        
        # attn caption information and encode it
        if self.word_attn:
            caption_attn = self.c_q_attn(caption, question)
            caption = torch.cat((caption, caption_attn), dim=2)
        caption_all, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        # attn dialog information and encode it
        if self.word_attn:
            dialog_attn = self.d_q_attn(dialog_history, question)
            dialog_history = torch.cat((dialog_history, dialog_attn), dim=2)
        dialog_all, encoded_dialog = self.dialog_encoder.forward(dialog_history)
        
        # read question
        question_all, encoded_question = self.question_encoder.forward(question, question_seq_helper)

        # form context
        if self.pure_text:
            fusion_set = [encoded_caption, encoded_dialog, encoded_question]
        elif self.use_i3d:
            fusion_set = [encoded_caption, encoded_dialog, encoded_question, encoded_vggish, encoded_flow, encoded_rgb]
        else:
            fusion_set = [encoded_caption, encoded_dialog, encoded_question, encoded_vggish]

        if self.text_fusion:
            context = torch.cat((caption_all, dialog_all), dim=1)

            # context fusion
            context_fusion = self.context_fusion(context, question_all, question_all)
            if self.cat_fusion:
                context_fusion_all, encoded_context_fusion = self.fusion_reader(torch.cat((context, context_fusion), dim=2))
            else:
                context_fusion_all, encoded_context_fusion = self.fusion_reader(context_fusion)
            context_fusion_all = context_fusion + context_fusion_all

            # self fusion
            self_fusion = self.self_fusion(context_fusion_all, context_fusion_all, context_fusion_all)
            if self.cat_fusion:
                self_fusion_all, encoded_self_fusion = self.self_reader(torch.cat((context_fusion_all, self_fusion), dim=2))
            else:
                self_fusion_all, encoded_self_fusion = self.self_reader(self_fusion)
            self_fusion_all = self_fusion + self_fusion_all

            fusion_set.append(encoded_context_fusion)
            fusion_set.append(encoded_self_fusion)
            # vgg fusion
            if self.video_fusion:
                vgg_fusion = self.vgg_fusion(vggish_all, self_fusion_all, self_fusion_all)
                if self.cat_fusion:
                    vgg_fusion_all, encoded_vgg_fusion = self.vgg_fusion_reader(torch.cat((vggish_all, vgg_fusion), dim=2))
                else:
                    vgg_fusion_all, encoded_vgg_fusion = self.vgg_fusion_reader(vgg_fusion)
                vgg_fusion_all = vgg_fusion + vgg_fusion_all
        
                fusion_all = vgg_fusion_all
                fusion_set.append(encoded_vgg_fusion)
            else:
                fusion_all = self_fusion_all
        else:
            fusion_all = vggish_all

        if self.fuse_attn:
            encoded_fusion = self.fusion(fusion_set)
        elif self.pure_text:
            encoded_fusion = self.fusion([encoded_caption, encoded_dialog, encoded_question])
        elif self.use_i3d:
            encoded_fusion = self.fusion([encoded_caption, encoded_dialog, encoded_question, encoded_vggish, encoded_flow, encoded_rgb])
        else:
            encoded_fusion = self.fusion([encoded_caption, encoded_dialog, encoded_question, encoded_vggish])

        if self.pure_text:
            features = [caption_all, dialog_all, question_all]
        elif self.use_i3d:
            features = [caption_all, dialog_all, question_all, vggish_all, rgb_all, flow_all]
        else:
            features = [caption_all, dialog_all, question_all, vggish_all]
        '''if self.text_fusion:
            features.append(context_fusion_all)
            features.append(self_fusion_all)
            if self.video_fusion:
                features.append(vgg_fusion_all)
        '''

        feature_vector = torch.cat(features, dim=1)

        if self.gen_attn == 'topdown':
            answer_indices, _ = self.answer_generator.generate(feature_vector, word_helper, fusion_vector=encoded_fusion)
        else:
            if self.text_fusion and self.attn_on_fusion:
                fusion_all = fusion_all
            elif self.attn_on_video:
                fusion_all = vggish_all
            elif self.attn_on_text:
                fusion_all = caption_all
            elif self.attn_on_dialog:
                fusion_all = dialog_all
            else:
                fusion_all = feature_vector
            answer_indices, _ = self.answer_generator.generate(fusion_all, encoded_fusion, word_helper)
        return answer_indices