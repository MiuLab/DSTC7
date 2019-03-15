from src.model.net.simple_model import SimpleModel
from src.model.net.fusion_model import FusionModel
from src.log import log
from src.util import utils
from src.util.WordHelper import WordHelper
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

import itertools
import pickle
from pathlib import Path
from nlgeval import NLGEval # Compute nlg scores
# TODO: Add validation BLEU score or crossentropy loss
# TODO: Load and continue training
# TODO: RMSProp
# TODO: Label smoothing
# TODO: Go to pytorch examples to check if there are tricks which I haven't implemented
# TODO: Save model along with some hyperparameter(e.g. lr, bidirectional)
# TODO: DialogModel abstract class
class DialogModel(object):
    def __init__(self, dataset, batch_size, shuffle, outputDir, modelType, word_helper, timestamp, context=1, restore_training=False):
        # Configuration
        log.logging.debug("[*] Loading NLGEval")
        self.nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
        log.logging.debug("[-] Done Loading NLGEval")
        self.start_lr = 5e-4
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.shuffle = shuffle
        self.outputDir = Path(outputDir)
        self.num_workers = 10
        self.use_cuda = False
        self.fusion_hidden_size = 256    
        self.timestamp = timestamp
        self.context = context
        self.restore_training = restore_training
        # Load word embedding
        log.logging.debug("[*] Loading Word Helper")
        self.word_helper = word_helper
        log.logging.debug("[-] Done Loading Word Helper")
        self.word_embed_size = self.word_helper.word_embed_size
        self.voc_size = self.word_helper.voc_size
        self.modelType = modelType
        log.logging.debug("[!] Using {} model".format(self.modelType))
        if self.modelType == "SimpleModel":
            self.model = SimpleModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, attention='fusion', dropout_rate=0.5, feature_set=0)
        elif self.modelType == 'TextModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru',  pure_text=True, word_attn=False, text_fusion=True, video_fusion=False, fuse_attn=True, attn_on_text=True, gen_attn='linear')
        elif self.modelType == 'ConvFusionModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=False, video_fusion=False, fuse_attn=False, fuse='conv11', gen_attn='none')
        elif self.modelType == 'ConvFusionLargeModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=False, video_fusion=False, fuse_attn=False, fuse='conv11_large', gen_attn='none')
        elif self.modelType == 'FusionModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=True, video_fusion=False, fuse_attn=True, gen_attn='linear')
        elif self.modelType == 'ConvFullModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=True, video_fusion=False, fuse_attn=True, fuse='conv11', gen_attn='linear')
        elif self.modelType == 'I3dFusionModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', use_i3d=True, word_attn=False, text_fusion=True, video_fusion=False, fuse_attn=True, gen_attn='linear')
        elif self.modelType == 'TextFusionModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=True, video_fusion=False, fuse_attn=False, attn_on_fusion=True, gen_attn='linear')
        elif self.modelType == 'TopDownModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=False, video_fusion=False, fuse_attn=False, gen_attn='topdown')
        elif self.modelType == 'AttentionModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=False, video_fusion=False, fuse_attn=False, gen_attn='linear')
        elif self.modelType == 'FusionWeightedModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=True, video_fusion=False, fuse_attn=True, gen_attn='none')
        elif self.modelType == 'VideoFusionWeightedModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=True, video_fusion=True, fuse_attn=True, gen_attn='none')
        elif self.modelType == 'TransformerModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='transformer', word_attn=False, text_fusion=False, video_fusion=False, fuse_attn=False, gen_attn='none')
        elif self.modelType == 'AttentionModel':
            self.model = FusionModel(self.fusion_hidden_size, self.word_embed_size, self.voc_size, dropout_rate=0.5, encoder='gru', word_attn=False, text_fusion=False, video_fusion=False, fuse_attn=False, gen_attn='linear')
        else:
            raise NotImplementedError
        
    # Copy from https://discuss.pytorch.org/t/adaptive-learning-rate/320/4
    def adjust_learning_rate(self, optimizer, epoch):
        if optimizer is None:
            return
        lr = self.start_lr * (0.1 ** ((epoch-1) // 30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def build(self):
        '''
            Set up optimizer
        ''' 
        # TODO: Remeber to add WordHelper class's parameter into it
        self.params_model = [p for p in self.model.parameters() if p.requires_grad]
        self.params_word_embedding = [p for p in self.word_helper.parameters() if p.requires_grad]
        # L2 penalty as https://discuss.pytorch.org/t/how-to-choose-a-suitable-weight-decay/1564 suggest
        self.optim_model = torch.optim.Adam(self.params_model, lr=self.start_lr, weight_decay=1e-4)
        #self.optim_model = torch.optim.Adamax(self.params_model, weight_decay=3e-7)
        self.optim_word_embedding = torch.optim.Adam(self.params_word_embedding, lr=self.start_lr, weight_decay=1e-4) if len(self.params_word_embedding) != 0 else None
        #self.optim_word_embedding = torch.optim.Adamax(self.params_word_embedding, weight_decay=3e-7) if len(self.params_word_embedding) != 0 else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.word_helper.word2index["<PAD>"]) # ignore padding loss
        #self.criterion_label = nn.CrossEntropyLoss(ignore_index=2)

    # TODO: Epoch initial information like epoch_loss, batch_loss
    def epoch_init(self):
        epoch_loss = 0.
        return epoch_loss

    def batch_init(self):
        batch_loss = 0.
        return batch_loss
            

    def fit(self, train_dataset, valid_dataset):
        # data_loader = torch.utils.data.DataLoader(train_dataset=train_dataset, 
        #                                           batch_size=self.batch_size, 
        #                                           shuffle=self.shuffle, 
        #                                           num_workers=self.num_workers,
        #                                           collate_fn=train_dataset.collate_fn, 
        #                                           drop_last=True)
        
        self.model.train()
        
        start_epoch = 1
        losses = []
        valid_losses = []
        min_valid_loss = float("inf")
        # Training Loop
        try:
            losses = []
            valid_losses = []
            min_valid_loss = float("inf")
            train_data_loader = train_dataset.batch_generator(self.batch_size, shuffle=self.shuffle, word_helper=self.word_helper, use_cuda=self.use_cuda, training=True)
            for epoch in range(start_epoch, 1000000):
                # Adjust learning rate
                self.adjust_learning_rate(self.optim_model, epoch)
                self.adjust_learning_rate(self.optim_word_embedding, epoch)

                print("Traing Epoch {}".format(epoch))
                # Initial loss
                epoch_loss = 0.
                #epoch_label_loss = 0.
            
                # ======================== Training =====================
                pbar = tqdm(train_data_loader, desc="Training", total=len(train_dataset) / self.batch_size)
                for batch_index, (i3d_flow, i3d_flow_seq_helper, 
                    i3d_rgb, i3d_rgb_seq_helper, 
                    vggish, vggish_seq_helper, 
                    caption, caption_seq_helper, 
                    batch_question, batch_question_seq_helper, 
                    batch_answer, batch_answer_seq_helper, batch_label) in enumerate(pbar):
                    
                    batch_size, _, _ = i3d_flow.size()
                    #
                    batch_loss = 0.
                    #batch_label_loss = 0.

                    last_batch_question, last_batch_answer = None, None
                    dialog_history = None
                    for i in range(10):
                        # Initialize dialog_history
                        if self.context:
                            if last_batch_answer is not None:
                                if self.context == 2:
                                    dialog_history = torch.cat((dialog_history, last_batch_answer), dim=1)
                                else:
                                    dialog_history = last_batch_answer
                            else:
                                dialog_history = torch.zeros(batch_size, 1, self.word_embed_size)
                        else:
                            dialog_history = torch.zeros(batch_size, self.fusion_hidden_size)
                            
                        if self.use_cuda:
                            dialog_history = dialog_history.cuda()

                        # Embed ith turn question
                        embed_batch_question = self.word_helper.embed(batch_question[i].cpu())
                        embed_batch_answer = self.word_helper.embed(batch_answer[i].cpu())
                        #embed_caption = self.word_helper.embed(caption)
                        if self.use_cuda:
                            embed_batch_question = embed_batch_question.cuda()
                            embed_batch_answer = embed_batch_answer.cuda()
                            #embed_caption = embed_caption.cuda()
                        
                        last_batch_question = embed_batch_question
                        last_batch_answer = embed_batch_answer
                         
                        # Generate ith answer
                        answer_word_prob, _ = self.model.forward(i3d_flow, i3d_flow_seq_helper, 
                                                                              i3d_rgb, i3d_rgb_seq_helper, 
                                                                              vggish, vggish_seq_helper,
                                                                              caption, caption_seq_helper, 
                                                                              embed_batch_question, batch_question_seq_helper[i], 
                                                                              dialog_history, 
                                                                              ground_truth_answer=embed_batch_answer, 
                                                                              ground_truth_answer_seq_helper=batch_answer_seq_helper[i],
                                                                              word_helper=self.word_helper)
                        # (B, T, voc_size) -> (*, voc_size)
                        answer_word_prob = answer_word_prob.reshape(-1, self.voc_size)
                        ground_truth_answer_indices = batch_answer[i][:, 1:].reshape(-1) # get rid of "<SOS>" token
                        
                        # (batch, )
                        label_target = batch_label[i].reshape(-1)

                        # Compute loss
                        gen_loss = self.criterion.forward(answer_word_prob, ground_truth_answer_indices)
                        #label_loss = self.criterion_label.forward(yn_prob, label_target)
                        loss = gen_loss
                        # Zero
                        self.model.zero_grad()
                        self.word_helper.zero_grad()
                        # Backward
                        loss.backward(retain_graph=False)
                        # clip gradient as https://github.com/pytorch/examples/blob/master/word_language_model/main.py suggest
                        torch.nn.utils.clip_grad_norm_(self.params_model, max_norm=5)
                        #torch.nn.utils.clip_grad_norm_(self.params_model, max_norm=5)
                        # Step
                        self.optim_model.step()
                        # If word embedding doesn't requires_grad
                        if self.optim_word_embedding is not None:
                            torch.nn.utils.clip_grad_norm_(self.params_word_embedding, max_norm=5)
                            #torch.nn.utils.clip_grad_norm_(self.params_word_embedding, max_norm=5)
                            self.optim_word_embedding.step() 
                        #
                        batch_loss += gen_loss.item()
                        #batch_label_loss += label_loss.item()

                    # divide by ten dialog turn
                    batch_loss /= 10
                    #batch_label_loss /= 10
                    # Incremental moving average
                    epoch_loss = epoch_loss * (batch_index / (batch_index+1)) + batch_loss / (batch_index+1)
                    #epoch_label_loss = epoch_label_loss * (batch_index / (batch_index+1)) + batch_label_loss / (batch_index+1)
                    
                    if batch_index == 200:
                        pbar.close()
                        break

                # ======================== Evaluate Validation Dataset =====================
                # TODO: too slow(maybe just choose some of them)
                with torch.no_grad():
                    valid_loss = self.eval_valid(valid_dataset)
                # Append epoch average loss
                losses.append(epoch_loss)
                valid_losses.append(valid_loss)
                # Log information
                print("Epoch Average Loss: {}".format(epoch_loss))
                #print('Epoch Average Label Loss {}'.format(epoch_label_loss))
                print("Epoch Valid Average Loss: {}".format(valid_loss))
                # Dump model and log every 2 epoch
                if valid_loss < min_valid_loss:
                    self.save()
                    min_valid_loss = valid_loss
                self.log(losses, valid_losses)

                # TODO: Only save model when valid CE loss or Bleu score improves
        except KeyboardInterrupt:
            pass
        # Log 
        self.log(losses, valid_losses)

    def eval_valid(self, dataset, stop_batch_idx=100):
        '''
            Compute Cross Entropy Loss on Validation Set
            To avoid high computation cost, we set stopping condition on batch_index, to avoid looping whole dataset
        '''
        self.model.eval()
        data_loader = dataset.batch_generator(self.batch_size, shuffle=True, word_helper=self.word_helper, use_cuda=self.use_cuda, training=False)
        pbar = tqdm(data_loader, desc="Evaluating Valid Set")
        # Average Loss of a dataset
        epoch_loss = 0.
        for batch_index, (i3d_flow, i3d_flow_seq_helper, 
                i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, 
                caption, caption_seq_helper, 
                batch_question, batch_question_seq_helper, 
                batch_answer, batch_answer_seq_helper, batch_label) in enumerate(pbar):
            # Get size
            batch_size, _, _ = i3d_flow.size()
            #
            batch_loss = 0.

            last_batch_question, last_batch_answer = None, None
            for i in range(10):
                # Initialize dialog_history
                if self.context:
                    if last_batch_answer is not None:
                        if self.context == 2:
                            dialog_history = torch.cat((dialog_history, last_batch_answer), dim=1)
                        else:
                            dialog_history = last_batch_answer
                    else:
                        dialog_history = torch.zeros(batch_size, 1, self.word_embed_size)
                else:
                    dialog_history = torch.zeros(batch_size, self.fusion_hidden_size)
                if self.use_cuda:
                    dialog_history = dialog_history.cuda()
        
                # Embed ith turn question
                embed_batch_question = self.word_helper.embed(batch_question[i].cpu())
                embed_batch_answer = self.word_helper.embed(batch_answer[i].cpu())
                if self.use_cuda:
                    embed_batch_question = embed_batch_question.cuda()
                    embed_batch_answer = embed_batch_answer.cuda()
                
                last_batch_question = embed_batch_question
                last_batch_answer = embed_batch_answer
                
                # Pass through ground truth
                answer_word_prob, _ = self.model.forward(i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                                   vggish, vggish_seq_helper, caption, caption_seq_helper, 
                                   embed_batch_question, batch_question_seq_helper[i], dialog_history,
                                   ground_truth_answer=embed_batch_answer, ground_truth_answer_seq_helper=batch_answer_seq_helper[i],
                                   word_helper=self.word_helper)
                # (B, T, voc_size) -> (*, voc_size)
                answer_word_prob = answer_word_prob.reshape(-1, self.voc_size)
                ground_truth_answer_indices = batch_answer[i][:, 1:].reshape(-1) # get rid of "<SOS>" token
                # Compute loss
                loss = self.criterion.forward(answer_word_prob, ground_truth_answer_indices)
                # Accumulate loss to batch_loss
                batch_loss += loss.item()
            # Divide batch_loss by 10 dialog turn
            batch_loss /= 10
            # Increment loss to epoch_loss
            epoch_loss = epoch_loss * (batch_index / (batch_index+1)) + batch_loss / (batch_index+1)
            # Stopping Condition
            if batch_index == stop_batch_idx:
                pbar.close()
                break
        self.model.train()
        # return this dataset's cross entropy loss
        return epoch_loss


    def predict(self, dataset, undisclose=False):
        torch.set_grad_enabled(False)
        self.model.eval()
        data_loader = dataset.batch_generator(self.eval_batch_size, shuffle=False, word_helper=self.word_helper, use_cuda=self.use_cuda, training=False)
        # TODO: beam search
        answers = [[] for i in range(10)]
        for batch_index, (i3d_flow, i3d_flow_seq_helper, 
                i3d_rgb, i3d_rgb_seq_helper, 
                vggish, vggish_seq_helper, 
                caption, caption_seq_helper, 
                batch_question, batch_question_seq_helper, 
                batch_answer, batch_answer_seq_helper, batch_label) in enumerate(tqdm(data_loader)):
            # Get size
            batch_size, _, _ = i3d_flow.size()
            
            last_batch_question, last_batch_answer = None, None
            # Two flows here:
            for i in range(10):
                # Initialize dialog_history
                if self.context:
                    if last_batch_answer is not None:
                        if self.context == 2:
                            dialog_history = torch.cat((dialog_history, last_batch_answer), dim=1)
                        else:
                            dialog_history = last_batch_answer
                    else:
                        dialog_history = torch.zeros(batch_size, 1, self.word_embed_size)
                else:
                    dialog_history = torch.zeros(batch_size, self.fusion_hidden_size)
                if self.use_cuda:
                    dialog_history = dialog_history.cuda()
                
                def get_vocab(indices):
                    vocab = np.array([[] for _ in range(batch_size)])
                    for indice in indices:
                        indice = indice.cpu().numpy() # (batch, seq)
                        vocab = np.concatenate((vocab, indice), axis=-1)
                    return vocab

                question_vocab = get_vocab(batch_question[:i+1])
                answer_vocab = get_vocab(batch_answer[:i])
                vocab = np.concatenate((question_vocab, answer_vocab), axis=-1)
                vocab_weight = torch.ones(batch_size, self.word_helper.voc_size)
                for idx in range(batch_size):
                    v = np.unique(vocab[idx, :])
                    vocab_weight[idx, v] = 1.05
                if self.use_cuda:
                    vocab_weight = vocab_weight.cuda()
                
                # Embed ith turn question
                embed_batch_question = self.word_helper.embed(batch_question[i].cpu())
                embed_batch_answer = self.word_helper.embed(batch_answer[i].cpu())
                if self.use_cuda:
                    embed_batch_question = embed_batch_question.cuda()
                    embed_batch_answer = embed_batch_answer.cuda()
                
                last_batch_question = embed_batch_question
                last_batch_answer = embed_batch_answer 
                
                answer_indices = self.model.generate(i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                                vggish, vggish_seq_helper, caption, caption_seq_helper, 
                                embed_batch_question, batch_question_seq_helper[i], dialog_history, self.word_helper, vocab_weight=vocab_weight)
                # (batch, timestep)
                if isinstance(answer_indices, torch.Tensor) == True:
                    answer_indices = answer_indices.tolist()
                elif isinstance(answer_indices, list):
                    pass
                # Apply indices2tokens function to every batch                    
                answer_tokens = list(map(self.word_helper.indices2tokens, answer_indices))
                # Extend a batch of answer tokens
                answers[i].extend(answer_tokens)
            
                '''# Pass through ground truth
                _, _ = self.model.forward(i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, 
                                   vggish, vggish_seq_helper, caption, caption_seq_helper, 
                                   embed_batch_question, batch_question_seq_helper[i], dialog_history,
                                   ground_truth_answer=embed_batch_answer, ground_truth_answer_seq_helper=batch_answer_seq_helper[i])
                '''
                # TODO: compute loss to ground truth (I think maybe we don't need it)
                #del _

        # Turn it pairwisely
        answers = list(zip(*answers))
        # compute nlgeval metrics
        metrics_dict = None
        if not undisclose:
            metrics_dict = self.compute_metrics(dataset, answers)
        self.model.train()
        return answers, metrics_dict
    
    def compute_metrics(self, dataset, answers):
        # unfold answers to create hypothesis
        hypothesis = list(itertools.chain(*answers))
        # make it join by spaces
        hypothesis = list(map(lambda x: " ".join(x), hypothesis))
        #
        references = [[]] # Only one reference
        for dialog_information in dataset.data["dialogs"]:
            if len(dialog_information["dialog"]) != 10:
                continue
            for qa_pair in dialog_information["dialog"]:
                # Append references
                references[0].append(qa_pair["answer"])
        # 
        metrics_dict = self.nlgeval.compute_metrics(references, hypothesis) 
        return metrics_dict

    def cuda(self):
        self.use_cuda = True
        self.model.cuda()
        
    def cpu(self):
        self.use_cuda = False
        self.model.cpu()

    def log(self, losses, valid_losses):
        '''
            Log important information
        '''
        fname = "loss_" + self.timestamp
        with (self.outputDir / "log" / fname).open("wb") as f:
            pickle.dump(losses, f)
        fname = "valid_losses_" + self.timestamp
        with (self.outputDir / "log" / fname).open("wb") as f:
            pickle.dump(valid_losses, f)

    # TODO: Save model hyperparamter along state_dict
    def save(self):
        fname = self.modelType + self.timestamp
        # 
        state_dict = self.model.state_dict()
        # Ensure everythings in cpu
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        # Save model
        torch.save(state_dict, str(self.outputDir / "parameter" / fname ) )
        # save wordhelper's embedding
        if self.optim_word_embedding is not None:
            self.word_helper.save_word_embedding()

    def load(self):
        fname = self.modelType + self.timestamp
        state_dict = torch.load(str(self.outputDir / "parameter" / fname))
        # deal with missing state key
        state = self.model.state_dict()
        state.update(state_dict)
        self.model.load_state_dict(state)
        # TODO: Fix adundant loading
        self.word_helper.load_word_embedding() 
    
    def load_log(self):
        '''
            Load log information so that we can keep training
        '''
        fname = "loss_" + self.timestamp
        with (self.outputDir / "log" / fname).open("rb") as f:
            training_losses = pickle.load(f)
        fname = "valid_losses_" + self.timestamp
        with (self.outputDir / "log" / fname).open("rb") as f:
            valid_losses = pickle.load(f)
        return training_losses, valid_losses