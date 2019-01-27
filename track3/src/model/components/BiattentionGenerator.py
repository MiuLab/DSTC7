import torch
from torch import nn
from collections import namedtuple
import torch.nn.functional as F

# A tuple for beam search
BeamNode = namedtuple('BeamNode', ['indices', 'hidden_state', 'log_prob', 'done'])

class Bilinear_pooling(nn.Module):
    def __init__(self, M, N, d):
        super(Bilinear_pooling, self).__init__()
        self.U = nn.Linear(N, d)
        self.V = nn.Linear(M, d)
        self.P = nn.Linear(d, 1)
    
    def forward(self, f, q):
        '''
        Args:
            f: (batch, seq, M)
            q: (batch, N)
        Return:
            mix: (batch, M)
        '''
        # (batch, N) -> (batch, d)
        uh = F.tanh(self.U(q))
        # (batch, seq, M) -> (batch, seq, d) -> (batch, d, seq)
        vf = F.tanh(self.V(f))
        vf = vf.permute(0, 2, 1) # transpose
        # (batch, d, 1) * (batch, d, seq) -> (batch, d, seq)
        biattn = uh.unsqueeze(2) * vf
        # (batch, d, seq) -> (batch, seq, d) -> (batch, seq)
        scores = self.P(biattn.permute(0, 2, 1))
        scores = F.softmax(scores.squeeze(2), 1)

        # (batch, 1, seq) * (batch, seq, M) -> (batch, 1, M)
        mix = torch.bmm(scores.unsqueeze(1), f)

        return mix.squeeze(1)

class GatedTanh(nn.Module):
    def __init__(self, in_size, out_size):
        super(GatedTanh, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.fc_gate = nn.Linear(in_size, out_size)

    def forward(self, input):
        y = F.tanh(self.fc(input))
        g = F.sigmoid(self.fc_gate(input))
        return y * g

class TopDown(nn.Module):
    def __init__(self, v_size, q_size, size):
        super(TopDown, self).__init__()
        self.gtanh = GatedTanh(v_size + q_size, size)
        self.fc = nn.Linear(size, 1)

    def forward(self, v, q):
        '''
        Args:
            v: (batch, seq, v_dim)
            q: (batch, q_dim)
        '''
        q = q.unsqueeze(1).expand(-1, v.size(1), -1)
        # (batch, seq, v_dim + q_dim)
        feature = torch.cat((v, q), dim=2)
        # (batch, seq, 1)
        scores = self.fc(self.gtanh(feature))
        attn = F.softmax(scores.squeeze(2), dim=-1)
        mix = torch.bmm(attn.unsqueeze(1), v)
        return mix.squeeze(1)

class LinearAttention(nn.Module):
    def __init__(self, v_size, q_size, size):
        super(LinearAttention, self).__init__()
        self.linear_q = nn.Linear(q_size, size)
        self.linear_v = nn.Linear(v_size, size)
    
    def forward(self, v, q):
        '''
        Args:
            q: (batch, q_size)
            v: (batch, seq, v_size)
        Return:
            mix : (batch, v_size)
        '''
        q_feature = F.relu(self.linear_q(q))
        v_feature = F.relu(self.linear_v(v))

        # (batch, seq, size) * (batch, size) -> (batch, seq)
        attn = torch.bmm(v_feature, q_feature.unsqueeze(2))
        attn = F.softmax(attn.squeeze(2), dim=-1)

        # (batch, 1, seq) * (batch, seq, v_size) -> (batch, 1, v_size)
        mix = torch.bmm(attn.unsqueeze(1), v)
        return mix.squeeze(1)

class PointerNet(nn.Module):
    def __init__(self,
                 context_size,
                 question_size,
                 ):
        super(PointerNet, self).__init__()
        self.fc_start = nn.Linear(question_size, context_size)

    def forward(self, context, question):
        weight = self.fc_start(question)
        # (batch, c_len, c_dim) * (batch, c_dim, 1) -> (batch, c_len)
        attn = context.bmm(weight.unsqueeze(2)).squeeze(2)
        scores = F.softmax(attn, -1)
        # (batch, 1, c_len) * (batch, c_len, c_dim) -> (batch, 1, c_dim)
        mix = scores.unsqueeze(1).bmm(context).squeeze(1) 
        return mix

class GatedLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super(GatedLinear, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.fc_gate = nn.Linear(in_size, out_size)

    def forward(self, input):
        y = self.fc(input)
        g = F.sigmoid(self.fc_gate(input))
        return y * g

class DynamicConv(nn.Module):
    def __init__(self, dim, kernel=5, h=16):
        super(DynamicConv, self).__init__()
        assert dim % h == 0
        self.dim = dim
        self.h = h
        self.kernel = kernel
        
        self.glu = GatedLinear(dim*2, dim)
        self.weights = nn.Parameter(torch.Tensor(h, kernel))
        nn.init.xavier_uniform_(self.weights)

        self.conv11 = nn.Conv1d(dim, 1, 1)
    def forward(self, context, question):
        '''
        Args:  
            question: (batch, dim)
            context: (batch, seq, dim)
        '''
        weights = self.weights.repeat(1, self.dim // self.h).view(self.dim, 1, self.kernel)
        # (batch, seq, c_dim + q_dim)
        question = question.unsqueeze(1).expand(-1, context.size(1), -1)
        feature = torch.cat((context, question), dim=2)
        feature = self.glu(feature)
        # input: (batch, seq, dim) -> (batch, dim, seq)
        input = feature.permute(0, 2, 1)
        input = F.conv1d(input, F.softmax(weights, dim=-1), padding=2, groups=self.dim)
        # -> (batch, 1, seq)
        attn = self.conv11(input)
        # weighted sum
        scores = F.softmax(attn, -1)
        mix = torch.bmm(scores, feature)
        return mix.squeeze(1)


class Biattention_Generator(nn.Module):
    # TODO: In training, we use teacher forcing. In testing, we use sampling.
    def __init__(self, input_size, hidden_size, voc_size, attention='bilinear'):
        super(Biattention_Generator, self).__init__()
        self.voc_size = voc_size
        self.attention = attention

        if attention == 'bilinear':
            self.attn = Bilinear_pooling(hidden_size, hidden_size, 256)
            self.gru_cell = nn.GRUCell(input_size=input_size + hidden_size, hidden_size=hidden_size)
            self.fc = nn.Linear(hidden_size*2, hidden_size)
            self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem
        elif attention == 'linear':
            self.attn = LinearAttention(hidden_size, hidden_size, 256)
            self.gru_cell = nn.GRUCell(input_size=input_size + hidden_size, hidden_size=hidden_size)
            self.fc = nn.Linear(hidden_size*2, hidden_size)
            self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem
        elif attention == 'pointer':
            self.attn = PointerNet(hidden_size, hidden_size)
            self.gru_cell = nn.GRUCell(input_size=input_size + hidden_size, hidden_size=hidden_size)
            self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem
        elif attention == 'topdown':
            self.attn = TopDown(hidden_size, hidden_size, 256)
            self.gru_cell = nn.GRUCell(input_size=input_size + hidden_size, hidden_size=hidden_size)
            self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem
        elif attention == 'd_conv':
            self.attn = DynamicConv(hidden_size)
            self.gru_cell = nn.GRUCell(input_size=input_size + hidden_size, hidden_size=hidden_size)
            self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem
        else:
            self.attn = None
            self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
            self.dense = nn.Linear(hidden_size, voc_size) # TODO: Solve too large voc size problem
        self.dropout = nn.Dropout(0.5)

    def forward(self, fusion_vector, hidden_vector, ground_truth_seq_helper, ground_truth_answer_input):
        '''
            ground_truth_answer_input: for teacher forcing usage, it contains input for GRUCell 
        '''
        # TODO: Schedule sampling
        batch_size, max_length, embed_size = ground_truth_answer_input.size()
        # Collect result
        result = []
        # Teacher forcing
        h = hidden_vector
        # Loop over timestep dimension
        for i in range(max_length-1):
            input = ground_truth_answer_input[:, i, :]
            if self.attn is not None:
                context = h
                input_attn = self.attn(fusion_vector, context)
                input = torch.cat((input, input_attn), dim=1)
            
            h = self.gru_cell.forward(input, h)

            '''if self.attention == 'linear':
                h_attn = self.fc(torch.cat((h, input_attn), dim=1))
                result.append(self.dense(h_attn))
            else:
                result.append(self.dense(h))
            '''
            result.append(self.dense(h))
            
             
        # Stack result along timestep dimension
        result = torch.stack(result, dim=1)
        # Return word probabilities vector and new dialog history
        return result

    def generate(self, fusion_vector, hidden_vector, word_helper, beam_size=5, batch_version=True):
        # (batch_size, fusion_hidden_size)
        batch_size, _ = hidden_vector.size()
        use_cuda = hidden_vector.is_cuda
        # set up first hidden state
        h = hidden_vector 
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
                if self.attn is not None:
                    context = torch.cat((input, h), dim=1)
                    input_attn = self.attn(fusion_vector, context)
                    input = torch.cat((input, input_attn), dim=1)
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
                result, h = self.batch_beam_search(input, EOSIndex, h, fusion_vector, word_helper, beam_size=beam_size)
            else:
                result, h = self.beam_search(input, EOSIndex, h, word_helper, beam_size=beam_size)
            
            return result, h

    # TODO: Add apply_function argument 
    def batch_beam_search(self, input, EOSIndex, h, fusion_vector, word_helper, beam_size):
        '''
            input: SOS tokens - (batch_size, embed_size)
            h: (batch_size, fusion_hidden_size)

        '''
        use_cuda = h.is_cuda
        batch_size, fusion_hidden_size = h.size()
        embed_size = input.size()[1]
        fusion_size = fusion_vector.size(2)
        # h_next is (batch_size, fusion_hidden_size)
        if self.attn is not None:
            context = h
            input_attn = self.attn(fusion_vector, context)
            input = torch.cat((input, input_attn), dim=1)

        h_next = self.gru_cell.forward(input, h)
        
        #if self.attention == 'linear' or self.attention == 'bilinear':
        #    h_attn = self.fc(torch.cat((h_next, input_attn), dim=1))
        #else:
        #    h_attn = h_next
        h_attn = h_next
        
        # (batch_size, fusion_hidden_size) -> (batch_size, voc_size)
        voc_logit = torch.nn.functional.log_softmax(self.dense(h_attn), dim=1)
        # (batch_size, 1, fusion_hidden_size)
        h_next = h_next.unsqueeze(dim=1)
        # (batch_size, beam_size, hidden_size)
        h_next = h_next.expand(-1, beam_size, -1)
        # expand fusion vector
        fusion_vector = fusion_vector.unsqueeze(dim=1)
        fusion_vector = fusion_vector.expand(-1, beam_size, -1, -1)
        fusion_vector = fusion_vector.reshape(batch_size*beam_size, -1, fusion_size)
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
            if self.attn is not None:
                context = h_next
                embed_attn = self.attn(fusion_vector, context)
                embed = torch.cat((embed, embed_attn), dim=1)

            h_next = self.gru_cell.forward(embed, h_next)

            #if self.attn is 'linear' or self.attn is 'bilinear':
            #    h_attn = self.fc(torch.cat((h_next, embed_attn), dim=1))
            #else:
            #    h_attn = h_next
            h_attn = h_next
        
            # Pass through dense layer
            voc_logit = self.dense.forward(h_attn)
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
