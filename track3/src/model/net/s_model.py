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

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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


class SModel(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 word_embed_size, 
                 voc_size, 
                 dropout_rate=0.2, 
                 gen_attn='linear',
                 text_aware=True,
                 text_attn=False,
                 decode_on_text=False,
                 video_attn=False):
        super(SModel, self).__init__()
        i3d_flow_dim, i3d_rgb_dim, vggish_dim = 2048, 2048, 128
        video_dim = i3d_flow_dim + i3d_rgb_dim + vggish_dim
        fusion_hidden_size = hidden_size * 2
        
        self.text_aware = text_aware
        if text_aware:
            self.flow_encoder = TextAwareReader(input_size=i3d_flow_dim,
                                                cond_size=fusion_hidden_size,
                                                out_size=1024,
                                                hidden_size=hidden_size)
            self.rgb_encoder = TextAwareReader(input_size=i3d_rgb_dim,
                                                cond_size=fusion_hidden_size,
                                                out_size=1024,
                                                hidden_size=hidden_size)
            self.vgg_encoder = TextAwareReader(input_size=vggish_dim,
                                                cond_size=fusion_hidden_size,
                                                out_size=vggish_dim,
                                                hidden_size=hidden_size)
        else:
            self.flow_encoder = DocReader(
                    input_size=i3d_flow_dim, hidden_size=hidden_size, 
                    num_layers=1, dropout=0, bidirectional=True)
            self.rgb_encoder = DocReader(
                    input_size=i3d_rgb_dim, hidden_size=hidden_size, 
                    num_layers=1, dropout=0, bidirectional=True)
            self.vgg_encoder = DocReader(
                    input_size=vggish_dim, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True
            )
            
        self.caption_encoder = DocReader(
                    input_size=word_embed_size, hidden_size=hidden_size, 
                    num_layers=1, dropout=0, bidirectional=True)
        self.question_encoder = DocReader(
                    input_size=word_embed_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        self.dialog_encoder = DocReader(
                    input_size=word_embed_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
        
        self.c_q_attn = WordAttention(word_embed_size)
        self.d_q_attn = WordAttention(word_embed_size)

        self.text_attn = text_attn
        if text_attn:
            self.fusion_reader = DocReader(
                    input_size=fusion_hidden_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
            self.self_reader = DocReader(
                    input_size=fusion_hidden_size, hidden_size=hidden_size,
                    num_layers=1, dropout=0, bidirectional=True)
            self.context_fusion = MultiHeadedAttention(8, fusion_hidden_size)
            self.self_fusion = MultiHeadedAttention(8, fusion_hidden_size)
            self.fusion = MultiChannelFusion(8)
        else:
            self.fusion = MultiChannelFusion(6)

        self.decode_on_text = decode_on_text
        if not decode_on_text:
            self.video_fusion = MultiChannelFusion(3)
        
        self.video_attn = video_attn
        if video_attn:
            self.video_reader = DocReader(
                input_size=fusion_hidden_size, hidden_size=hidden_size,
                num_layers=1, dropout=0, bidirectional=True)
            self.video_attn = MultiHeadedAttention(8, fusion_hidden_size)

        # answer generator
        self.answer_generator = Biattention_Generator(
                        input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                        voc_size=voc_size, attention=gen_attn)
        # y/n predict
        #self.yn_predict = nn.Linear(fusion_hidden_size, 3)

        # dropouts
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog, 
                ground_truth_answer, ground_truth_answer_seq_helper, word_helper=None):
        # dropout
        i3d_flow = self.dropout(i3d_flow)
        i3d_rgb = self.dropout(i3d_rgb)
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog = self.dropout(dialog)

        # attn caption information and encode it
        caption_attn = self.c_q_attn(caption, question)
        caption_all, caption_encoded = self.caption_encoder.forward(caption_attn)
        
        # attn dialog information and encode it
        dialog_attn = self.d_q_attn(dialog, question)
        dialog_all, dialog_encoded = self.dialog_encoder.forward(dialog_attn)
        
        # read question and cat all text information
        question_all, question_encoded = self.question_encoder.forward(question)
        context = torch.cat((caption_all, dialog_all), dim=1)

        # video reader
        if self.text_aware:
            flow_all, flow_encoded = self.flow_encoder.forward(i3d_flow, caption_encoded)
            rgb_all, rgb_encoded = self.rgb_encoder.forward(i3d_rgb, caption_encoded)
            vgg_all, vgg_encoded = self.vgg_encoder.forward(vggish, caption_encoded)
        else:
            flow_all, flow_encoded = self.flow_encoder.forward(i3d_flow)
            rgb_all, rgb_encoded = self.rgb_encoder.forward(i3d_rgb)
            vgg_all, vgg_encoded = self.vgg_encoder.forward(vggish)

        if self.text_attn:
            # context fusion
            context_fusion = self.context_fusion(context, question_all, question_all)
            context_fusion_all, context_fusion_encoded = self.fusion_reader(context_fusion)
            context_fusion_all = context_fusion + context_fusion_all

            # self fusion
            self_fusion = self.self_fusion(context_fusion_all, context_fusion_all, context_fusion_all)
            self_fusion_all, self_fusion_encoded = self.self_reader(self_fusion)
            self_fusion_all = self_fusion + self_fusion_all
            
            fusion = self.fusion(flow_encoded, 
                                rgb_encoded, 
                                vgg_encoded, 
                                caption_encoded, 
                                dialog_encoded, 
                                question_encoded,
                                context_fusion_encoded,
                                self_fusion_encoded
                                )
        else:
            fusion = self.fusion(flow_encoded, 
                                rgb_encoded, 
                                vgg_encoded, 
                                caption_encoded, 
                                dialog_encoded, 
                                question_encoded,
                                )
        if self.decode_on_text:
            attn_seq = self_fusion_all
        else:
            # fuse all video
            batch = flow_all.size(0)
            video_len = flow_all.size(1)
            video_fusion = self.video_fusion(flow_all.contiguous().view(batch, -1), 
                                            rgb_all.contiguous().view(batch, -1), 
                                            vgg_all.contiguous().view(batch, -1))
            if self.video_attn:
                video = video_fusion.view(batch, video_len, -1)
                video = self.video_attn(video, self_fusion_all, self_fusion_all)
                video_all, video_encoded = self.video_reader(video)
                attn_seq = video + video_all
            else:
                attn_seq = video_fusion.view(batch, video_len, -1)

        answer_word_probabilities = self.answer_generator.forward(attn_seq, fusion, 
            ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer)

        return answer_word_probabilities, None

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog, word_helper, vocab_weight=None):
       # dropout
        i3d_flow = self.dropout(i3d_flow)
        i3d_rgb = self.dropout(i3d_rgb)
        vggish = self.dropout(vggish)
        caption = self.dropout(caption)
        question = self.dropout(question)
        dialog = self.dropout(dialog)

        # attn caption information and encode it
        caption_attn = self.c_q_attn(caption, question)
        caption_all, caption_encoded = self.caption_encoder.forward(caption_attn)
        
        # attn dialog information and encode it
        dialog_attn = self.d_q_attn(dialog, question)
        dialog_all, dialog_encoded = self.dialog_encoder.forward(dialog_attn)
        
        # read question and cat all text information
        question_all, question_encoded = self.question_encoder.forward(question)
        context = torch.cat((caption_all, dialog_all), dim=1)

        # video reader
        if self.text_aware:
            flow_all, flow_encoded = self.flow_encoder.forward(i3d_flow, caption_encoded)
            rgb_all, rgb_encoded = self.rgb_encoder.forward(i3d_rgb, caption_encoded)
            vgg_all, vgg_encoded = self.vgg_encoder.forward(vggish, caption_encoded)
        else:
            flow_all, flow_encoded = self.flow_encoder.forward(i3d_flow)
            rgb_all, rgb_encoded = self.rgb_encoder.forward(i3d_rgb)
            vgg_all, vgg_encoded = self.vgg_encoder.forward(vggish)

        if self.text_attn:
            # context fusion
            context_fusion = self.context_fusion(context, question_all, question_all)
            context_fusion_all, context_fusion_encoded = self.fusion_reader(context_fusion)
            context_fusion_all = context_fusion + context_fusion_all

            # self fusion
            self_fusion = self.self_fusion(context_fusion_all, context_fusion_all, context_fusion_all)
            self_fusion_all, self_fusion_encoded = self.self_reader(self_fusion)
            self_fusion_all = self_fusion + self_fusion_all
            
            fusion = self.fusion(flow_encoded, 
                                rgb_encoded, 
                                vgg_encoded, 
                                caption_encoded, 
                                dialog_encoded, 
                                question_encoded,
                                context_fusion_encoded,
                                self_fusion_encoded
                                )
        else:
            fusion = self.fusion(flow_encoded, 
                                rgb_encoded, 
                                vgg_encoded, 
                                caption_encoded, 
                                dialog_encoded, 
                                question_encoded,
                                )
        if self.decode_on_text:
            attn_seq = self_fusion_all
        else:
            # fuse all video
            batch = flow_all.size(0)
            video_len = flow_all.size(1)
            video_fusion = self.video_fusion(flow_all.contiguous().view(batch, -1), 
                                            rgb_all.contiguous().view(batch, -1), 
                                            vgg_all.contiguous().view(batch, -1))
            if self.video_attn:
                video = video_fusion.view(batch, video_len, -1)
                video = self.video_attn(video, caption_all, caption_all)
                video_all, video_encoded = self.video_reader(video)
                attn_seq = video + video_all
            else:
                attn_seq = video_fusion.view(batch, video_len, -1)

        # Generate answers
        answer_indices, _ = self.answer_generator.generate(attn_seq, fusion, word_helper)
    
        return answer_indices
        