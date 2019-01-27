import torch
import math
from torch import nn
from . import simple_model
from ..components.SimpleGRUEncoder import Simple_GRU_Encoder
from ..components.FeatureEncoder import I3D_Flow_Encoder, I3D_RGB_Encoder, VGGish_Encoder, Caption_Encoder
from ..components.QuestionEncoder import Question_Encoder

class AttentionModel(nn.Module):
    def __init__(self, fusion_hidden_size, word_embed_size, voc_size):
        super(AttentionModel, self).__init__()
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
        self.caption_encoder = Caption_Encoder(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                    num_layers=1, dropout=0, bidirectional=False)
        self.question_encoder = Question_Encoder(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size,
                    num_layers=1, dropout=0, bidirectional=False)
        self.answer_generator = Attention_AnswerGenerator(
                    input_size=word_embed_size, hidden_size=fusion_hidden_size, 
                    voc_size=voc_size)

    def forward(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, 
                ground_truth_answer, ground_truth_answer_seq_helper):
        i3d_flow_output, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        i3d_rgb_output, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        vggish_output, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        caption_output, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        question_output, encoded_question = self.question_encoder.forward(question, question_seq_helper)

        answer_word_probabilities, new_dialog_history = self.answer_generator.forward(i3d_flow_output, i3d_rgb_output, 
            vggish_output, caption_output, question_output, dialog_history, 
            ground_truth_seq_helper=ground_truth_answer_seq_helper, ground_truth_answer_input=ground_truth_answer)

        return answer_word_probabilities, new_dialog_history

    def generate(self, i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                question, question_seq_helper, dialog_history, word_helper):
        # Encode past information
        i3d_flow_output, encoded_i3d_flow = self.i3d_flow_encoder.forward(i3d_flow, i3d_flow_seq_helper)
        i3d_rgb_output, encoded_i3d_rgb = self.i3d_rgb_encoder.forward(i3d_rgb, i3d_rgb_seq_helper)
        vggish_output, encoded_vggish = self.vggish_encoder.forward(vggish, vggish_seq_helper)
        caption_output, encoded_caption = self.caption_encoder.forward(caption, caption_seq_helper)
        question_output, encoded_question = self.question_encoder.forward(question, question_seq_helper)
        # Generate answers
        answer_indices, _ = self.answer_generator.generate(i3d_flow_output, i3d_rgb_output, 
            vggish_output, caption_output, question_output, dialog_history, word_helper)
        
        return answer_indices
# As https://arxiv.org/pdf/1701.03126.pdf describe
class AttentionFusion(nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()
    
    def forward(self, i3d_flow, i3d_rgb, vggish, caption, question, dialog_history_vector, answer_hidden, beam_size=None):
        '''
            all input are (batch_size, timesteps, fusion_hidden_size)
            dialog_history_vector: (batch_size, fusion_hidden_size)
            answer_hidden: (batch_size, fusion_hidden_size)
        '''
        fusion_hidden_size = i3d_flow.size()[2]
        # Support beam_search decoding, deal with answer_hidden is (batch_size * beam_size, fusion_hidden_size)
        if beam_size is not None:
            # NOTE: expand would not increase memory
            # (batch_size, timesteps, fusion_hidden_size) -> (batch_size, 1, timesteps, fusion_hidden_size)
            # -> (batch_size, beam_size, timesteps, fusion_hidden_size) 
            # -> (batch_size * beam_size, timesteps, fusion_hidden_size)
            timesteps = i3d_flow.size()[1]
            i3d_flow = i3d_flow.unsqueeze(dim=1).expand(-1, beam_size, -1, -1).reshape(-1, timesteps, fusion_hidden_size)
            timesteps = i3d_rgb.size()[1]
            i3d_rgb = i3d_rgb.unsqueeze(dim=1).expand(-1, beam_size, -1, -1).reshape(-1, timesteps, fusion_hidden_size)
            timesteps = vggish.size()[1]
            vggish = vggish.unsqueeze(dim=1).expand(-1, beam_size, -1, -1).reshape(-1, timesteps, fusion_hidden_size)
            timesteps = caption.size()[1]
            caption = caption.unsqueeze(dim=1).expand(-1, beam_size, -1, -1).reshape(-1, timesteps, fusion_hidden_size)
            timesteps = question.size()[1]
            question = question.unsqueeze(dim=1).expand(-1, beam_size, -1, -1).reshape(-1, timesteps, fusion_hidden_size)
            # (batch_size, fusion_hidden_size) -> (batch_size, 1, fusion_hidden_size) -> (batch_size, beam_size, fusion_hidden_size)
            dialog_history_vector = dialog_history_vector.unsqueeze(dim=1).expand(-1, beam_size, -1).reshape(-1, fusion_hidden_size)
        
        # Inspired by Attention is All You Need
        denominator = math.sqrt(fusion_hidden_size)
        # (batch_size, 1, fusion_hidden_size) * (batch_size, timesteps, fusion_hidden_size) ->(sum) (batch_size, timesteps, 1) ->(softmax over dim=1)
        i3d_flow_weight = nn.functional.softmax(torch.sum(answer_hidden.unsqueeze(dim=1) * i3d_flow, dim=2, keepdim=True) / denominator, dim=1)
        # (batch_size, timesteps, 1) * (batch_size, timesteps, fusion_hidden_size) -> (batch_size, fusion_hidden_size)
        i3d_flow_context = torch.sum(i3d_flow_weight * i3d_flow, dim=1)
        #
        i3d_rgb_weight = nn.functional.softmax(torch.sum(answer_hidden.unsqueeze(dim=1) * i3d_rgb, dim=2, keepdim=True) / denominator, dim=1)
        i3d_rgb_context = torch.sum(i3d_rgb_weight * i3d_rgb, dim=1)
        #
        vggish_weight = nn.functional.softmax(torch.sum(answer_hidden.unsqueeze(dim=1) * vggish, dim=2, keepdim=True) / denominator, dim=1)
        vggish_context = torch.sum(vggish_weight * vggish, dim=1)
        # 
        caption_weight = nn.functional.softmax(torch.sum(answer_hidden.unsqueeze(dim=1) * caption, dim=2, keepdim=True) / denominator, dim=1)
        caption_context = torch.sum(caption_weight * caption, dim=1)
        #
        question_weight = nn.functional.softmax(torch.sum(answer_hidden.unsqueeze(dim=1) * question, dim=2, keepdim=True) / denominator, dim=1)
        question_context = torch.sum(question_weight * question, dim=1)
        # Attention based multimodal fusion
        features_context = self.attention_based_multimodal_fusion(i3d_flow_context, i3d_rgb_context, vggish_context, caption_context, question_context, dialog_history_vector, answer_hidden)
        # Add features information and answer hidden state information together
        all_context = features_context + answer_hidden
        return all_context

    
    def attention_based_multimodal_fusion(self, i3d_flow_context, i3d_rgb_context, vggish_context, caption_context, question_context, dialog_history_vector, answer_hidden):
        '''
            features are: (batch_size, hidden_size)
            dialog_history_vector: (batch_size, hidden_size)
        '''
        fusion_hidden_size = i3d_flow_context.size()[1]
        denominator = math.sqrt(fusion_hidden_size)
        # (batch_size, 6, fusion_hidden_size)
        features = torch.stack([i3d_flow_context, i3d_rgb_context, vggish_context, caption_context, question_context, dialog_history_vector], dim=1)
        # (batch_size, 1, fusion_hidden_size) * (batch_size, 6, fusion_hidden_size) -> (batch_size, 6, 1)
        features_weight = torch.sum(answer_hidden.unsqueeze(dim=1) * features, dim=2, keepdim=True)
        # (batch_size, 6, 1) softmax over dim=1
        features_weight = nn.functional.softmax(features_weight / denominator, dim=1)
        # Apply weighed sum: (batch_size, 6, 1) * (batch_size, 6, fusion_hidden_size) -> (batch_size, fusion_hidden_size)
        features_context = torch.sum(features_weight * features, dim=1)
        return features_context


# TODO: Double check if code is right or not
class Attention_AnswerGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, voc_size):
        super(Attention_AnswerGenerator, self).__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.fusion = AttentionFusion()
        self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem
    
    def forward(self, i3d_flow_output, i3d_rgb_output, 
                vggish_output, caption_output, question_output, 
                dialog_history_vector, ground_truth_seq_helper, ground_truth_answer_input):
        
        #
        batch_size, max_length, embed_size = ground_truth_answer_input.size()
        use_cuda = i3d_flow_output.is_cuda 
        # Collect result
        result = []
        h_list = []
        # 
        h = torch.zeros(batch_size, self.hidden_size).cuda() if use_cuda else torch.zeros(batch_size, self.hidden_size)
        # Autoregressive
        for i in range(max_length-1):
            h = self.gru_cell.forward(ground_truth_answer_input[:, i, :], h)
            # ===================== Attention ==============================
            all_context = self.fusion.forward(i3d_flow_output, i3d_rgb_output, vggish_output, caption_output, question_output, dialog_history_vector, h)
            # Collect hidden state
            h_list.append(h)
            result.append(self.dense(all_context))
        
        # Stack result along timestep dimension
        h_list = torch.stack(h_list, dim=1) # (batch_size, max_length-1, hidden_size)
        result = torch.stack(result, dim=1)

        # Select last hidden state
        last_hidden_indices = ground_truth_seq_helper.seq_len - 2 # minus 2 is because max_length-1
        # (batch_size, ) -> (batch_size, 1) -> (batch_size, 1, 1) -> (batch_size, 1, hidden_size))
        last_hidden_indices = last_hidden_indices.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, -1, self.hidden_size)
        # Select over timestep dimension: (batch_size, max_length-1, hidden_size) -> (batch_size, 1, hidden_size)
        new_dialog_history = torch.gather(h_list, dim=1, index=last_hidden_indices)

        return result, new_dialog_history

    def generate(self, i3d_flow_output, i3d_rgb_output, 
            vggish_output, caption_output, question_output, 
            dialog_history_vector, word_helper, beam_size=5, batch_version=True):
        # (batch_size, fusion_hidden_size)
        batch_size, _ = dialog_history_vector.size()
        use_cuda = dialog_history_vector.is_cuda
        # Initial
        h = torch.zeros(batch_size, self.hidden_size).cuda() if use_cuda else torch.zeros(batch_size, self.hidden_size)
        # Set up input
        input = word_helper.embed([word_helper.word2index["<SOS>"]] * batch_size)
        # Cuda
        if use_cuda:
            input = input.cuda()
        # 
        EOSIndex = word_helper.word2index["<EOS>"]
        if batch_version:
            result, h = self.batch_beam_search(input, EOSIndex, h, i3d_flow_output, i3d_rgb_output, vggish_output, caption_output, question_output, dialog_history_vector, word_helper, beam_size=beam_size)
        else:
            raise NotImplementedError

        return result, h


    # TODO: Modulelize batch_beam_search
    def batch_beam_search(self, input, EOSIndex, h, i3d_flow_output, i3d_rgb_output, vggish_output, caption_output, question_output, dialog_history_vector, word_helper, beam_size):
        '''
            input: SOS tokens - (batch_size, embed_size)
            h: (batch_size, fusion_hidden_size)

        '''
        use_cuda = h.is_cuda
        batch_size, fusion_hidden_size = h.size()
        embed_size = input.size()[1]
        # h_next is (batch_size, fusion_hidden_size)
        h_next = self.gru_cell.forward(input, h)
        # (batch_size, fusion_hidden_size) -> (batch_size, voc_size)
        all_context = self.fusion.forward(i3d_flow_output, i3d_rgb_output, vggish_output, caption_output, question_output, dialog_history_vector, h_next)
        voc_logit = torch.nn.functional.log_softmax(self.dense(all_context), dim=1)
        # (batch_size, 1, fusion_hidden_size)
        h_next = h_next.unsqueeze(dim=1)
        # (batch_size, beam_size, hidden_size)
        h_next = h_next.expand(-1, beam_size, -1)
        # (batch_size, voc_size) -> (batch_size, beam_size)
        previous_log_prob, saved_indices = torch.topk(voc_logit, k=beam_size, dim=1)
        # (batch_size, beam_size, 1)
        saved_indices = saved_indices.unsqueeze(dim=2)
        # Record if beams are done
        beam_done = torch.zeros_like(saved_indices).squeeze(dim=2).byte() # (batch_size, beam_size)
        eos_tensor = torch.full_like(beam_done, fill_value=EOSIndex).long() # (batch_size, beam_size)
        # Loop over timestep
        for i in range(49):
            # Retrieve last word indices from (batch_size, beam_size, [timesteps]) -> (batch_size, beam_size)
            previous_indices = saved_indices[:, :, -1]
            # (batch_size, beam_size) -> (batch_size, beam_size, embed_size)
            embed = word_helper.embed(previous_indices, use_cuda=use_cuda)
            # (batch_size, beam_size, ?) -> (batch_size * beam_size, ?)
            embed = embed.reshape(-1, embed_size)
            h_next = h_next.reshape(-1, fusion_hidden_size)
            # Pass all beams to gru_cell. h_next is (batch_size * beam_size, hidden_size)
            h_next = self.gru_cell.forward(embed, h_next)
            # all_context: (batch_size * beam_size, hidden_size)
            all_context = self.fusion.forward(i3d_flow_output, i3d_rgb_output, vggish_output, caption_output, question_output, dialog_history_vector, h_next, beam_size=beam_size)
            # Pass through dense layer
            voc_logit = self.dense.forward(all_context)
            # (batch_size, beam_size, voc_size)
            voc_logit = voc_logit.reshape(batch_size, beam_size, self.voc_size)
            # log sofmax over voc_size dimension
            voc_logit = torch.nn.functional.log_softmax(voc_logit, dim=2)
            # broadcast add previous log prob: (batch_size, beam, 1) + (batch_size, beam, voc_size) 
            # NOTE: Trick: Mask EOS beam's logit, if that beam is already done, add nothing
            added_voc_logit = previous_log_prob.unsqueeze(dim=2) + voc_logit * (~beam_done).float().unsqueeze(dim=2)
            # (batch_size, beam, voc_size) -> (batch_size, beam * voc_size)
            added_voc_logit = added_voc_logit.reshape(batch_size, -1)
            # Choose top k beams: (batch_size, beam * voc_size) -> (batch_size, beam)
            new_log_prob, indices = torch.topk(added_voc_logit, k=beam_size, dim=1)
            # (batch_size, beam)
            pointer_to_previous_beam = indices / self.voc_size # long type division
            true_voc_indices = indices % self.voc_size # long type mod
            # Reshape to (batch_size, beam_size, fusion_hidden_size)
            h_next = h_next.reshape(batch_size, beam_size, fusion_hidden_size)
            # 
            timesteps = saved_indices.size()[-1]
            # ================= Choose Beams ======================
            # select beam's indices: (batch_size, beam_size, [timesteps]) -> (batch_size, beam_size, [timesteps])
            saved_indices = torch.gather(saved_indices, dim=1, index=pointer_to_previous_beam.unsqueeze(dim=2).expand(-1, -1, timesteps))
            # select beam's hidden state: (batch_size, beam_size, fusion_hidden_size) -> (batch_size, beam_size, fusion_hidden_size)
            # NOTE: Even if a beam reach EOSIndex, it will still pass through gru_cell. That means it will contain garbage information
            h_next = torch.gather(h_next, dim=1, index=pointer_to_previous_beam.unsqueeze(dim=2).expand(-1, -1, fusion_hidden_size))
            # select beam's done: (batch_size, beam_size) -> (batch_size, beam_size)
            beam_done = torch.gather(beam_done, dim=1, index=pointer_to_previous_beam)

            # ============= Prepare Things to Next Step ===========
            # (batch_size, beam_size, [timesteps] + 1)
            saved_indices = torch.cat([saved_indices, true_voc_indices.unsqueeze(dim=2)], dim=2)
            # Pass to next
            previous_log_prob = new_log_prob
            # ============= Check if EOS ==========================
            # (batch_size, beam_size)
            is_done = torch.eq(true_voc_indices, eos_tensor)
            # Record beam_done, if that beam has done
            beam_done |= is_done 
            
            # Check if all done
            if torch.sum(beam_done) == batch_size * beam_size:
                break

        # (batch_size, beam_size) -> (batch_size, 1) 
        best_beam = torch.argmax(previous_log_prob, dim=1, keepdim=True)
        timesteps = saved_indices.size()[2]
        # (batch_size, 1, 1)
        best_beam = best_beam.unsqueeze(dim=2)
        # Select over beam dimension: (batch_size, 1, timesteps)
        result = torch.gather(saved_indices, dim=1, index=best_beam.expand(-1, -1, timesteps))
        # (batch_size, timesteps)
        result = result.squeeze(dim=1)
        # (batch_size, 1, fusion_hidden_size)
        batch_h = torch.gather(h_next, dim=1, index=best_beam.expand(-1, -1, fusion_hidden_size))
        batch_h = batch_h.squeeze(dim=1)
        # BUG: batch_h may contain garbage information
        return result, batch_h


