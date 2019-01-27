import torch
import math
from torch import nn
from . import simple_model
from ..components.AnswerGenerator import Answer_Generator

# TODO: Attend on video stream and question sequence
class ModalAttentionModel(simple_model.SimpleModel):
    # Change answer_generator to Attention version
    def __init__(self, fusion_hidden_size, word_embed_size, voc_size):
        super().__init__(fusion_hidden_size, word_embed_size, voc_size)
        self.answer_generator = AttentionAnswerGenerator(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                    voc_size=voc_size)

class AttentionFusion(nn.Module):
    # TODO: bilinear attend
    def __init__(self, attend_method):
        super(AttentionFusion, self).__init__()
        attend_method_set = set(["dot", "bilinear"])
        self.attend_method = attend_method
        assert attend_method in attend_method_set
        
    def forward(self, encoded_i3d_flow, encoded_i3d_rgb, 
                encoded_vggish, encoded_caption, encoded_question, dialog_history_vector):
        # (batch_size, fusion_hidden_size) * (batch_size, fusion_hidden_size) -> (batch_size, 1)
        # TODO: softmax weight, divide by square root dimension
        # TODO: Adding encoded_question is a little weird?
        batch_size, fusion_hidden_size = encoded_i3d_flow.size()
        # (batch_size, 5)
        weight = torch.cat([torch.sum(encoded_i3d_flow * encoded_question, dim=1, keepdim=True), 
                            torch.sum(encoded_i3d_rgb * encoded_question, dim=1, keepdim=True),
                            torch.sum(encoded_vggish * encoded_question, dim=1, keepdim=True),
                            torch.sum(encoded_caption * encoded_question, dim=1, keepdim=True),
                            torch.sum(dialog_history_vector * encoded_question, dim=1, keepdim=True)], dim=1)
        # Divide by sqrt(fusion_hidden_size) as Attention is all you need suggest
        weight /= math.sqrt(fusion_hidden_size)
        # Take softmax over feature candidates dimension
        weight = torch.nn.functional.softmax(weight, dim=1)
        # (batch_size, 5, fusion_hidden_size)
        features = torch.stack([encoded_i3d_flow, encoded_i3d_rgb, encoded_vggish, encoded_caption, dialog_history_vector], dim=1)
        # weighted sum (batch_size, 5, 1) * (batch_size, 5, fusion_hidden_size) -> (batch_size, fusion_hidden_size) 
        context = torch.sum(weight.unsqueeze(dim=-1) * features, dim=1)
        # Add question
        context += encoded_question
        return context



class AttentionAnswerGenerator(Answer_Generator):
    def __init__(self, input_size, hidden_size, voc_size):
        super().__init__(input_size, hidden_size, voc_size)
        self.fusion = AttentionFusion(attend_method="dot")