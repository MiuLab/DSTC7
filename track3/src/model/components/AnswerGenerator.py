import torch
from torch import nn
from collections import namedtuple
import torch.nn.functional as F
import random

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

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, fusion_candidate_size=6):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv1 = nn.Conv1d(fusion_candidate_size, fusion_candidate_size*2, 1)
        self.conv2 = nn.Conv1d(fusion_candidate_size*2, 1, 1)
    
    def forward(self, *inputs):
        input = torch.stack(inputs, dim=1)
        mix = self.conv2(self.conv1(input))
        return mix.squeeze(1)

class MultiplyAttention(nn.Module):
    def __init__(self, dim):
        super(MultiplyAttention, self).__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, c, q):
        '''
        Args:
            c: (batch, dim)
            q: (batch, q_len, dim)
        Return:
            mix: (batch, dim)
        '''
        batch = c.size(0)
        q_len = q.size(1)
        c = c.unsqueeze(1)

        c_feature = F.relu(self.linear(c))
        q_feature = F.relu(self.linear(q))
        # (batch, 1, dim) * (batch, q_len, dim) -> (batch, 1, q_len)
        attn = torch.bmm(c_feature, q_feature.transpose(1, 2))
        attn = F.softmax(attn.view(-1, q_len), -1).view(batch, -1, q_len)
        # (batch, 1, q_len) * (batch, q_len, dim) -> (batch, 1, dim)
        mix = torch.bmm(attn, q)
        return mix.squeeze()

# A tuple for beam search
BeamNode = namedtuple('BeamNode', ['indices', 'hidden_state', 'log_prob', 'done'])
feature_dict = {0: 6, 1: 4, 2: 3, 3:2, 4:2}

class Answer_Generator(nn.Module):
    # TODO: In training, we use teacher forcing. In testing, we use sampling.
    def __init__(self, input_size, hidden_size, voc_size, attention='fusion', feature_set=0):
        super(Answer_Generator, self).__init__()
        self.voc_size = voc_size
        self.feature_set = feature_set
        self.attention = attention
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        if attention == 'fusion':
            self.fusion = Fusion(feature_dict[feature_set])
            #self.fusion = DepthwiseSeparableConv()
        elif attention == 'multiply':
            self.fusion = MultiplyAttention(dim = hidden_size)
        self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem

    def forward(self, encoded_i3d_flow, encoded_i3d_rgb, 
                encoded_vggish, encoded_caption, encoded_question, 
                dialog_history_vector, ground_truth_seq_helper, ground_truth_answer_input, word_helper=None):
        '''
            ground_truth_answer_input: for teacher forcing usage, it contains input for GRUCell 
        '''
        # TODO: Schedule sampling
        batch_size, max_length, embed_size = ground_truth_answer_input.size()
        use_cuda = ground_truth_answer_input.is_cuda
        # Fuse information together
        if self.attention == 'fusion':
            if self.feature_set == 0:
                fusion_vector = self.fusion.forward(encoded_i3d_flow, encoded_i3d_rgb, 
                                    encoded_vggish, encoded_caption, encoded_question, dialog_history_vector)
            elif self.feature_set == 1:
                fusion_vector = self.fusion.forward(encoded_vggish, encoded_caption, encoded_question, dialog_history_vector)
            elif self.feature_set == 2:
                fusion_vector = self.fusion.forward(encoded_caption, encoded_question, dialog_history_vector)
            elif self.feature_set == 3:
                fusion_vector = self.fusion.forward(encoded_question, dialog_history_vector)
            elif self.feature_set == 4:
                fusion_vector = self.fusion.forward(encoded_question, encoded_caption)
        elif self.attention == 'multiply':
            feature_vector = torch.stack((encoded_i3d_flow, encoded_i3d_rgb, 
                                        encoded_vggish, encoded_caption, 
                                        encoded_question, dialog_history_vector),
                                        dim=1)
            fusion_vector = self.fusion.forward(encoded_question, feature_vector)
        # Collect result
        result = []
        # Teacher forcing
        h = fusion_vector
        # Loop over timestep dimension
        last_output = None
        for i in range(max_length-1):
            h = self.gru_cell.forward(ground_truth_answer_input[:, i, :], h)
            # Collect hidden state
            result.append(self.dense(h))

        # Stack result along timestep dimension
        result = torch.stack(result, dim=1)
        
        # Return word probabilities vector and new dialog history
        return result, fusion_vector

    def generate(self, encoded_i3d_flow, encoded_i3d_rgb, 
            encoded_vggish, encoded_caption, encoded_question, 
            dialog_history_vector, word_helper, beam_size=5, batch_version=True, vocab_weight=None):
        # (batch_size, fusion_hidden_size)
        batch_size, _ = dialog_history_vector.size()
        use_cuda = dialog_history_vector.is_cuda
        # fuse information together
        if self.attention == 'fusion':
            if self.feature_set == 0:
                fusion_vector = self.fusion.forward(encoded_i3d_flow, encoded_i3d_rgb, 
                                    encoded_vggish, encoded_caption, encoded_question, dialog_history_vector)
            elif self.feature_set == 1:
                fusion_vector = self.fusion.forward(encoded_vggish, encoded_caption, encoded_question, dialog_history_vector)
            elif self.feature_set == 2:
                fusion_vector = self.fusion.forward(encoded_caption, encoded_question, dialog_history_vector)
            elif self.feature_set == 3:
                fusion_vector = self.fusion.forward(encoded_question, dialog_history_vector)
            elif self.feature_set == 4:
                fusion_vector = self.fusion.forward(encoded_question, encoded_caption)
        elif self.attention == 'multiply':
            feature_vector = torch.stack((encoded_i3d_flow, encoded_i3d_rgb, 
                                        encoded_vggish, encoded_caption, 
                                        encoded_question, dialog_history_vector),
                                        dim=1)
            fusion_vector = self.fusion.forward(encoded_question, feature_vector) 
        # set up first hidden state
        h = fusion_vector 
        # set up done vector 
        done = torch.zeros(batch_size).byte()
        # Maximum 100 words, TODO: maybe better way?
        input = word_helper.embed([word_helper.word2index["<SOS>"]] * batch_size)
        if use_cuda:
            input = input.cuda()
            done = done.cuda()
        
        EOSIndex = word_helper.word2index["<EOS>"]
        # Greedy search
        if beam_size == 1:
            result = []
            for i in range(50):
                # Forward 
                h = self.gru_cell.forward(input, h)
                # Get logit (batch_size, voc_size)
                voc_distribution_logit = self.dense(h)
                # softmax 
                voc_distribution = torch.nn.functional.softmax(voc_distribution_logit, dim=1)
                # TODO: sampling
                # Take argmax over voc_size dimension: (batch_size, )
                indices = torch.argmax(voc_distribution, dim=1)
                # embed output and assign it to input (batch_size, embed_size)
                input = word_helper.embed(indices, use_cuda)
                # Or operation
                done |= (indices == EOSIndex)
                # Append result
                result.append(indices)
                # check if all done
                if torch.sum(done) == batch_size:
                    break
            # Stack along time dimension
            result = torch.stack(result, dim=1)
            # NOTE: `h` may contain some garbage information
            # TODO: Fix h
            # TODO: return sequence length
            return result, h
        else:
            # Return result (batch, timestamp), hidden state(batch, fusion_hidden_size)
            if batch_version:
                result, h = self.batch_beam_search(input, EOSIndex, h, word_helper, beam_size=beam_size, vocab_weight=vocab_weight)
            else:
                result, h = self.beam_search(input, EOSIndex, h, word_helper, beam_size=beam_size)
            
            return result, h

    # TODO: Add apply_function argument 
    def batch_beam_search(self, input, EOSIndex, h, word_helper, beam_size, vocab_weight=None):
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
        voc_logit = torch.nn.functional.log_softmax(self.dense(h_next), dim=1)
        if vocab_weight is not None:
            voc_logit = voc_logit * vocab_weight
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
            # Pass all beams to gru_cell
            h_next = self.gru_cell.forward(embed, h_next)
            # Pass through dense layer
            voc_logit = self.dense.forward(h_next)
            # (batch_size, beam_size, voc_size)
            voc_logit = voc_logit.reshape(batch_size, beam_size, self.voc_size)
            # log sofmax over voc_size dimension
            voc_logit = torch.nn.functional.log_softmax(voc_logit, dim=2)
            if vocab_weight is not None:
                voc_logit  = voc_logit * vocab_weight.unsqueeze(1)
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

    # TODO: What if first generated word is EOS?
    def beam_search(self, input, EOSIndex, h, word_helper, beam_size):
        '''
            input: SOS tokens - (batch_size, embed_size)
            h: (batch_size, fusion_hidden_size)

        '''
        use_cuda = h.is_cuda
        batch_size, _ = h.size()
        result = []
        batch_h = []
        for i in range(batch_size):
            # Save beam, each beam contain a tuple of (indices = [], now hidden state, accumulated log probability, if_done)
            beam_branches = []
            # Loop over timestep dimension
            for t in range(50):
                # Initial
                if t == 0:
                    # (1, hidden_size)
                    h_next = self.gru_cell(input[i:i+1], h[i:i+1])
                    # (1, voc_size)
                    voc_logit = torch.nn.functional.log_softmax(self.dense(h_next), dim=1)
                    # topk -> (1, beam_size)
                    log_prob, indices = torch.topk(voc_logit, k=beam_size, dim=1)
                    # Save information into beam branches   
                    for j in range(beam_size):
                        beam_node = BeamNode(indices=[ indices[0, j] ] , 
                                            hidden_state=h_next, 
                                            log_prob=log_prob[0, j], done=False)
                        beam_branches.append(beam_node)
                else:
                    # For computation efficiency, we stack beam together to compute
                    # (beam_size, )
                    beam_indices = []
                    beam_h_prev = []
                    beam_prob_prev = []
                    beam_done = []
                    for j in range(beam_size):
                        '''
                            indices_prev: a list contain indices along that beam
                            h_prev: (1, fusion_hidden_size)
                            log_prob_prev: log probability until now
                        '''
                        indices_prev, h_prev, log_prob_prev, done_prev = beam_branches[j]
                        # Append indices_prev last indices
                        beam_indices.append(indices_prev[-1])
                        beam_h_prev.append(h_prev)
                        beam_prob_prev.append(log_prob_prev)
                        beam_done.append(done_prev)
                    # (beam_size, )
                    beam_done = torch.tensor(beam_done).cuda() if use_cuda else torch.tensor(beam_done)
                    # Embed -> (beam_size, embed_size)
                    embed_beam_indices = word_helper.embed(beam_indices, use_cuda=use_cuda)
                    # (beam_size, fusion_hidden_size)
                    beam_h_prev = torch.cat(beam_h_prev, dim=0)
                    # (beam_size, )
                    beam_prob_prev = torch.stack(beam_prob_prev, dim=0)
                    # feed it to gru cell h_next: (beam_size, fusion_hidden_size)
                    h_next = self.gru_cell(embed_beam_indices, beam_h_prev)
                    # dense (beam_size, voc_size)
                    voc_logit = nn.functional.log_softmax(self.dense(h_next), dim=1)
                    # NOTE: I mask out EOS beam here
                    # (beam_size, 1) + (beam_size, voc_size) -> (batch_size, voc_size)
                    added_voc_logit = beam_prob_prev.unsqueeze(dim=1) + voc_logit * ((~beam_done).unsqueeze(dim=1).float())
                    # flatten and take top k
                    # (beam_size, ), (beam_size, )
                    log_prob, indices = torch.topk(added_voc_logit.view(-1), k=beam_size, dim=0)
                    # create new beam
                    new_beam_branches = []
                    # Loop over new beam's indices
                    for j in range(beam_size):
                        # Transform to true beam index it belongs (actually, is division's quotient)
                        true_beam_idx = indices[j] / self.voc_size
                        true_index = indices[j] % self.voc_size 
                        
                        indices_prev, h_prev, log_prob_prev, done = beam_branches[true_beam_idx]
                        # If done, then don't take action, because all the things should remain the same
                        if done == True:
                            new_beam_branches.append(beam_branches[true_beam_idx])
                            continue
                        # Append true index behind its corresponding beam
                        new_indices = indices_prev + [true_index]
                        # Create new beam
                        # Check if that beam is done
                        done = True if true_index == EOSIndex else False
                        # NOTE: h_next and added_voc_logit should retrieve true_beam_idx. added_voc_logit[true_beam_idx, true_index] == log_prob[j]
                        new_beam_node = BeamNode(new_indices, h_next[true_beam_idx:true_beam_idx+1], added_voc_logit[true_beam_idx, true_index], done)
                        # Append new beam
                        new_beam_branches.append(new_beam_node)
                        

                    # Replace beam brances by new beam branches
                    beam_branches = new_beam_branches
                    # Check if all beam is done
                    if sum(beam_node.done for beam_node in beam_branches) == beam_size:
                        break

            # choose best beam in beam branches
            beam = max(beam_branches, key=lambda x: x.log_prob)
            # save that beam into result
            indices, hidden_state, _, _ = beam
            result.append(indices)
            batch_h.append(hidden_state)
        
        assert(isinstance(result, list))
        # Stack along batch_size dimension -> (batch, timestep)
        batch_h = torch.cat(batch_h, dim=0)
        return result, batch_h

if __name__ == '__main__':
    model = TopDownAttention(input_dim=300, 
                            h1_dim=256, 
                            h2_dim=256,
                            feature_dim=256)
    input = torch.Tensor(32, 300)
    v = torch.Tensor(32, 6, 256)
    h1 = torch.Tensor(32, 256)
    h2 = torch.Tensor(32, 256)

    model(input, v, h1, h2)
