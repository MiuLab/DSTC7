from torch.utils.data import Dataset
import torch
from pathlib import Path
from collections import namedtuple
from src.util.WordHelper import WordHelper
from src.util.SequenceHelper import SequenceHelper
from src.log import log
import numpy as np
import json
import itertools
import random
from tqdm import tqdm

# NOTE: Increase file descriptor
# See https://github.com/pytorch/pytorch/issues/973 for more information
import resource
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard_limit))

class AVSD_Dataset(Dataset):
    '''
        Load Audio, Video and Text data.
    '''
    def __init__(self, data_base_path, timestamp, word_helper, data_category="train", undisclose=False, video_upsample=False):
        # Declare member variable
        self.multimodal_dialogs = None # A list containing samples
        self.caption_max_len = None 
        self.questions_max_len = None
        self.answers_max_len = None
        self.data_path = Path(data_base_path) # 
        self.timestamp = timestamp
        self.data_category = data_category
        self.video_upsample = video_upsample
        self.audio_visual_path = self.data_path / "Audio-Visual-Feature"
        self.i3d_flow_path = self.audio_visual_path / "i3d_flow"
        self.i3d_rgb_path = self.audio_visual_path / "i3d_rgb"
        self.vggish_path = self.audio_visual_path / "vggish"
        self.word_embedding_path = self.data_path / "word_embedding"
        if not undisclose:
            self.text_json_path = self.data_path / "textdata" / (self.data_category + "_set.json")
        else:
            assert data_category == "test", "You should use \'test\'"
            self.text_json_path = self.data_path / "textdata" / "test_set4DSTC7-AVSD.json"
        
        # Open text json file
        with self.text_json_path.open() as f:
            self.data = data = json.load(f)
 
        image_ids = [] # A list contains all image_id
        captions = [] # A List of indices list 
        questions = [] # A list of question indices list
        answers = [] # A list of answer indices list 
        labels = []
        log.logging.debug("[*] Processing Text Data")
        for i, dialog_information in tqdm(enumerate(data["dialogs"])):
            caption_indices = word_helper.tokens2indices(dialog_information["caption"].split(" "))
            # TODO: Save invalid dialog turns (Method: Pad all questions to 20)
            # NOTE: Throw away invalid dialog turns
            
            dialog = dialog_information["dialog"]
            
            if len(dialog) > 10:
                print("Dialog length exceed 10!")
                continue
            # Pad dialog to lenght 10
            if len(dialog) < 10:
                if data_category == 'test':
                    dialog = self.pad_question_and_answer(dialog)
                else:
                    continue

            dialog_questions = []
            dialog_answers = []
            dialog_labels = []
            for qa_pair in dialog:
                # Encode question, answer to indices
                question = qa_pair['question'].split(' ')
                answer = qa_pair['answer'].split(' ')
                
                question_indices = word_helper.tokens2indices(question)
                answer_indices = word_helper.tokens2indices(answer)
                label = self.answer_label(answer)

                dialog_questions.append(question_indices)
                dialog_answers.append(answer_indices)
                dialog_labels.append(label)
            # Append each data sample
            image_ids.append(dialog_information["image_id"])
            captions.append(caption_indices)
            assert len(dialog_questions) == 10 and len(dialog_answers) == 10
            questions.append(dialog_questions)
            answers.append(dialog_answers)
            labels.append(dialog_labels)

        log.logging.debug("[-] Done Processing Text Data")
        log.logging.debug("[*] Padding")
        # NOTE: Pad all captions to the longest length in the captions' data
        captions_seq_len = WordHelper.getSequenceLength(captions)
        captions_max_len = max(captions_seq_len)
        captions = [word_helper.pad(sequence, captions_max_len) for sequence in captions]
        # NOTE: questions, answers are nested lists
        # Pad all question to same length
        questions_seq_len = [WordHelper.getSequenceLength(dialog_questions) for dialog_questions in questions] 
        questions_max_len = max(itertools.chain(*questions_seq_len)) # Unfold first level and take maximum
        for i in range(len(questions)):
            for j in range(len(questions[i])):
                questions[i][j] = word_helper.pad(questions[i][j], questions_max_len)

        # Pad all answers to same length
        answers_seq_len = [WordHelper.getSequenceLength(dialog_answers) for dialog_answers in answers]
        answers_max_len = max(itertools.chain(*answers_seq_len)) # Unfold first level
        for i in range(len(answers)):
            for j in range(len(answers[i])):
                answers[i][j] = word_helper.pad(answers[i][j], answers_max_len)
        log.logging.debug("[-] Done Padding")
        # Prepare data
        # multimodal_dialogs contain each sample as a tuple 
        self.caption_max_len = captions_max_len
        self.questions_max_len = questions_max_len
        self.answers_max_len = answers_max_len
        self.multimodal_dialogs = list(zip(image_ids, 
                                        captions, captions_seq_len,
                                        questions, questions_seq_len,
                                        answers, answers_seq_len,
                                        labels))
        
    def __len__(self):
        return len(self.multimodal_dialogs)

    def __getitem__(self, index):
        # 
        image_id, caption, caption_seq_len, question_list, questions_seq_len, answer_list, answers_seq_len, labels = self.multimodal_dialogs[index]
        # NOTE: Load audio visual feature here
        i3d_flow, i3d_rgb, vggish = self._load_audio_visual_feature(image_id)
        return i3d_flow, i3d_rgb, vggish, caption, caption_seq_len, question_list, questions_seq_len, answer_list, answers_seq_len, labels
        
    def _load_audio_visual_feature(self, image_id):
        '''
            Input: 
                image_id
            Output:
                3 numpy arrays
        '''
        image_id_npy = image_id + ".npy"
        i3d_flow = np.load(self.i3d_flow_path / image_id_npy)
        i3d_rgb = np.load(self.i3d_rgb_path / image_id_npy)
        vggish = np.load(self.vggish_path / image_id_npy)
        
        len_flow, len_rgb, len_vgg = i3d_flow.shape[0], i3d_rgb.shape[0], vggish.shape[0]
        if len_flow == len_rgb - 1:
            len_rgb -= 1
            i3d_rgb = i3d_rgb[:len_rgb]
        if self.video_upsample:
            idx = np.round(np.linspace(0, len_vgg-1, len_flow)).astype(np.int)
            vggish = vggish[idx]
        
        return i3d_flow, i3d_rgb, vggish
    
    @staticmethod
    def answer_label(text):
        '''
        Args:
            text: list of str
        '''
        if text[0] == 'yes':
            return 0
        elif text[0] == 'no':
            return 1
        else:
            return 2

    @staticmethod
    def pad_question_and_answer(dialog):
        '''
            This function is to pad dstc testing json to 10 question answer pairs
            Input:
                one instance of dialog 
            Output:
                padded dialog with length == 10
        '''
        PAD_QUESTION = "PAD_QUESTION"
        PAD_ANSWER = "PAD_ANSWER"
        assert len(dialog) < 10, "This dialog is bigger than 10 which should not appear in this function"
        pad_qa_pair = dict([("question", PAD_QUESTION), ("answer", PAD_ANSWER)])
        pad_length = 10 - len(dialog)

        padded_dialog = dialog + [pad_qa_pair for _ in range(pad_length)]
        return padded_dialog

    @staticmethod
    def pad_audio_visual_feature_in_batch(batch_features):
        """
            Input:
                A list of features
            Output:
                padded result, sequence lengths, maximum length
        """
        seq_len = list(map(len, batch_features))
        max_len = max(seq_len)
        padded_batch_features = [np.pad(seq, [(0, max_len - len(seq)), (0, 0)], mode='constant') for seq in batch_features]
        padded_batch_features = np.stack(padded_batch_features)
        return padded_batch_features, seq_len, max_len

    @staticmethod
    def create_sequence_mask(seq_len, max_len):
        batch_size = len(seq_len)
        mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(seq_len):
            mask[i, :length] = 1 # binary mask
        return mask 

    # TODO: Do less things in collate_fn to avoid file descriptors leakage
    def collate_fn(self, data):
        '''
            Input:
                data: A list of tuple 
                (

                )
            Output:
                All features, questions, answers and their corresponding SequenceHelper class
        '''
        i3d_flow, i3d_rgb, vggish, \
        caption, caption_seq_len, \
        question, question_seq_len, answer, answer_seq_len, label = zip(*data)
        
        # Pad i3d_flow, i3d_rgb, vggish
        i3d_flow, i3d_flow_seq_len, i3d_flow_max_len = AVSD_Dataset.pad_audio_visual_feature_in_batch(i3d_flow)
        i3d_rgb, i3d_rgb_seq_len, i3d_rgb_max_len = AVSD_Dataset.pad_audio_visual_feature_in_batch(i3d_rgb)
        vggish, vggish_seq_len, vggish_max_len = AVSD_Dataset.pad_audio_visual_feature_in_batch(vggish)
        # Features mask
        i3d_flow_mask = AVSD_Dataset.create_sequence_mask(i3d_flow_seq_len, i3d_flow_max_len)
        i3d_rgb_mask = AVSD_Dataset.create_sequence_mask(i3d_rgb_seq_len, i3d_rgb_max_len)
        vggish_mask = AVSD_Dataset.create_sequence_mask(vggish_seq_len, vggish_max_len)
        caption_mask = AVSD_Dataset.create_sequence_mask(caption_seq_len, self.caption_max_len)

        # Questions, Answers sequence mask
        batch_question = [None] * 10
        batch_answer = [None] * 10
        batch_label = [None] * 10
        batch_question_seq_len = [None] * 10
        batch_answer_seq_len = [None] * 10
        batch_question_mask = [None] * 10
        batch_answer_mask = [None] * 10
        # Collect sequence length for each dialog turn(e.g. collect 1st question together, collect 2nd....)
        for i in range(10):
            #
            batch_question[i] = torch.LongTensor([ten_turn_question[i] for ten_turn_question in question])
            batch_answer[i] = torch.LongTensor([ten_turn_answer[i] for ten_turn_answer in answer])
            batch_label[i] = torch.LongTensor([ten_turn_label[i] for ten_turn_label in label])
            # Collect ith length
            batch_question_seq_len[i] = [seq_len[i] for seq_len in question_seq_len] # sequence lengths in ith question
            batch_answer_seq_len[i] = [seq_len[i] for seq_len in answer_seq_len] # in ith answer
            batch_question_mask[i] = AVSD_Dataset.create_sequence_mask(batch_question_seq_len[i], self.questions_max_len)
            batch_answer_mask[i] = AVSD_Dataset.create_sequence_mask(batch_answer_seq_len[i], self.answers_max_len)
        
        # Create sequence helper
        i3d_flow_seq_helper = SequenceHelper(i3d_flow_seq_len, i3d_flow_max_len, i3d_flow_mask)
        i3d_rgb_seq_helper = SequenceHelper(i3d_rgb_seq_len, i3d_rgb_max_len, i3d_rgb_mask)
        vggish_seq_helper = SequenceHelper(vggish_seq_len, vggish_max_len, vggish_mask)
        caption_seq_heler = SequenceHelper(caption_seq_len, self.caption_max_len, caption_mask)
        batch_question_seq_helper = [SequenceHelper(batch_question_seq_len[i], self.questions_max_len, batch_question_mask[i]) for i in range(10)]
        batch_answer_seq_helper = [SequenceHelper(batch_answer_seq_len[i], self.answers_max_len, batch_answer_mask[i]) for i in range(10)]

        # Turn all things into torch.Tensor type
        i3d_flow = torch.from_numpy(i3d_flow).float()
        i3d_rgb = torch.from_numpy(i3d_rgb).float()
        vggish = torch.from_numpy(vggish).float()
        caption = torch.LongTensor(caption)

        return \
            i3d_flow, i3d_flow_seq_helper, \
            i3d_rgb, i3d_rgb_seq_helper, \
            vggish, vggish_seq_helper, \
            caption, caption_seq_heler, \
            batch_question, batch_question_seq_helper, \
            batch_answer, batch_answer_seq_helper, \
            batch_label
    def batch_generator(self, batch_size, shuffle, word_helper, use_cuda, training):
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)
        else:
            pass

        num_batch = len(self) // batch_size
        # Deal with remainder case
        if len(self) % batch_size != 0:
            num_batch += 1
        
        # Inifitie loop for training 
        while True:
            for batch_index in range(num_batch):
                # NOTE: python's list can deal with out-of-range indexing like `list[1:100000000]` 
                data = [self[index] for index in indices[batch_index * batch_size: (batch_index + 1) * batch_size]]
                # 
                i3d_flow, i3d_flow_seq_helper, i3d_rgb, i3d_rgb_seq_helper, \
                vggish, vggish_seq_helper, caption, caption_seq_helper, \
                batch_question, batch_question_seq_helper, \
                batch_answer, batch_answer_seq_helper, batch_label = self.collate_fn(data)
                
                # embed 
                caption = word_helper.embed(caption)
                # Use cuda
                if use_cuda:
                    i3d_flow = i3d_flow.cuda()
                    i3d_flow_seq_helper = i3d_flow_seq_helper.cuda()
                    i3d_rgb = i3d_rgb.cuda()
                    i3d_rgb_seq_helper = i3d_rgb_seq_helper.cuda()
                    vggish = vggish.cuda()
                    vggish_seq_helper = vggish_seq_helper.cuda()
                    caption = caption.cuda()
                    caption_seq_helper = caption_seq_helper.cuda()
                    batch_question = [b_q.cuda() for b_q in batch_question]
                    batch_question_seq_helper = [seq_helper.cuda() for seq_helper in batch_question_seq_helper]
                    batch_answer = [b_a.cuda() for b_a in batch_answer]
                    batch_answer_seq_helper = [seq_helper.cuda() for seq_helper in batch_answer_seq_helper]
                    batch_label = [b_l.cuda() for b_l in batch_label]
                    

                yield i3d_flow, i3d_flow_seq_helper, \
                    i3d_rgb, i3d_rgb_seq_helper, \
                    vggish, vggish_seq_helper, \
                    caption, caption_seq_helper, \
                    batch_question, batch_question_seq_helper, \
                    batch_answer, batch_answer_seq_helper, \
                    batch_label
            # if not training, break
            if not training:
                break




        
if __name__ == "__main__":
    dataset = AVSD_Dataset("./AVSD_Jim/data", "train")
    from IPython import embed
    embed()
    
        



