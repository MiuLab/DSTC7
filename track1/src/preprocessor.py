import json
import logging
import spacy
from multiprocessing import Pool
from dataset import DSTC7Dataset
from tqdm import tqdm


class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embeddings):
        self.nlp = spacy.load('en_core_web_sm',
                              disable=['tagger', 'ner', 'parser', 'textcat'])
        self.tokenized_train = None
        self.tokenized_valid = None
        self.embeddings = embeddings

        if 'speaker1' not in embeddings.word_dict:
            embeddings.add('speaker1')
        if 'speaker2' not in embeddings.word_dict:
            embeddings.add('speaker2')

    def tokenize(self, sentence):
        return [token.text
                for token in self.nlp(sentence)]

    def sentence_to_indices(self, sentence):
        return [self.embeddings.to_index(token)
                for token in self.tokenize(sentence)]

    def get_dataset(self, data_path, n_workers=16, **preprocess_args):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        logging.info('loading dataset...')
        with open(data_path) as f:
            dataset = json.load(f)

        logging.info('preprocessing data...')

        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_dataset,
                                              [batch, preprocess_args])
                # self.preprocess_dataset(batch, preprocess_args)

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()

        return DSTC7Dataset(processed)

    def preprocess_dataset(self, dataset, preprocess_args):
        """
        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset):
            processed.append(self.preprocess(sample, **preprocess_args))

        return processed

    def preprocess(self, data, cat=True, testing=False):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['id'] = data['example-id']

        # process messages so far
        processed['context'] = []
        processed['speaker'] = []
        for message in data['messages-so-far']:
            processed['context'].append(
                self.sentence_to_indices(message['utterance'].lower())
            )
            speaker = message['speaker'].replace('student', 'participant_1') \
                                        .replace('advisor', 'participant_2')
            processed['speaker'].append(
                ['participant_1', 'participant_2'].index(speaker)
            )

        # process options
        processed['options'] = []
        processed['option_ids'] = []
        if not testing:
            for option in data['options-for-correct-answers']:
                processed['options'].append(
                    self.sentence_to_indices(option['utterance'].lower())
                )
                processed['option_ids'].append(option['candidate-id'])

        for option in data['options-for-next']:
            processed['options'].append(
                self.sentence_to_indices(option['utterance'].lower())
            )
            processed['option_ids'].append(option['candidate-id'])

        if not testing:
            processed['n_corrects'] = len(data['options-for-correct-answers'])
        else:
            processed['n_corrects'] = 0

        if cat:
            context = []
            utterance_ends = []
            assert len(processed['speaker']) == len(processed['context'])
            for speaker, utterance in zip(processed['speaker'],
                                          processed['context']):
                context.append(
                    self.embeddings.to_index('speaker{}'.format(speaker + 1))
                )
                context += utterance
                utterance_ends.append(len(context) - 1)

            processed['context'] = context
            processed['utterance_ends'] = utterance_ends

        if 'profile' in data:
            profile = data['profile']['Courses']
            priors = [
                self.embeddings.to_index(
                    course['offering'].split('-')[0].lower()
                )
                for course in profile['Prior']
            ]
            suggested = [
                self.embeddings.to_index(
                    course['offering'].split('-')[0].lower()
                )
                for course in profile['Suggested']
            ]
            processed['prior'] = [
                1 if w in priors else 0
                for w in processed['context']
            ]
            processed['suggested'] = [
                1 if w in suggested else 0
                for w in processed['context']
            ]
            processed['option_prior'] = [
                [1 if w in priors else 0
                 for w in opt]
                for opt in processed['options']
            ]
            processed['option_suggested'] = [
                [1 if w in suggested else 0
                 for w in opt]
                for opt in processed['options']
            ]
        else:
            processed['prior'] = [
                0 for w in processed['context']
            ]
            processed['suggested'] = [
                0 for w in processed['context']
            ]
            processed['option_prior'] = [
                [0 for w in opt]
                for opt in processed['options']
            ]
            processed['option_suggested'] = [
                [0 for w in opt]
                for opt in processed['options']
            ]

        return processed


class RolePreprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embeddings):
        self.nlp = spacy.load('en_core_web_sm',
                              disable=['tagger', 'ner', 'parser', 'textcat'])
        self.tokenized_train = None
        self.tokenized_valid = None
        self.embeddings = embeddings

        if 'speaker1' not in embeddings.word_dict:
            embeddings.add('speaker1')
        if 'speaker2' not in embeddings.word_dict:
            embeddings.add('speaker2')

    def tokenize(self, sentence):
        return [token.text
                for token in self.nlp(sentence)]

    def sentence_to_indices(self, sentence):
        return [self.embeddings.to_index(token)
                for token in self.tokenize(sentence)]

    def get_dataset(self, data_path, n_workers=16, **preprocess_args):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        logging.info('loading dataset...')
        with open(data_path) as f:
            dataset = json.load(f)

        logging.info('preprocessing data...')

        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_dataset,
                                              [batch, preprocess_args])
                # self.preprocess_dataset(batch, preprocess_args)

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()

        return DSTC7Dataset(processed)

    def preprocess_dataset(self, dataset, preprocess_args):
        """
        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in dataset:
            processed.append(self.preprocess(sample, **preprocess_args))

        return processed

    def preprocess(self, data, cat=True):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['id'] = data['example-id']

        # process messages so far
        processed['context'] = []
        processed['speaker'] = []
        for message in data['messages-so-far']:
            processed['context'].append(
                self.sentence_to_indices(message['utterance'].lower())
            )
            speaker = message['speaker'].replace('student', 'participant_1') \
                                        .replace('advisor', 'participant_2')
            processed['speaker'].append(
                ['participant_1', 'participant_2'].index(speaker)
            )

        processed['context1'] = []
        processed['context2'] = []
        for speaker, utterance in zip(processed['speaker'], processed['context']):
            processed['context{}'.format(speaker+1)].append(utterance)

        processed['context1_masked'] = []
        processed['context2_masked'] = []
        for speaker, utterance in zip(processed['speaker'], processed['context']):
            other_speaker = (speaker + 1) % 2
            masked_utterance = [
                self.embeddings.to_index('<unk>')
                for _ in range(len(utterance))
            ]
            processed['context{}_masked'.format(speaker+1)].append(utterance)
            processed['context{}_masked'.format(other_speaker+1)].append(masked_utterance)

        # process options
        processed['options'] = []
        processed['option_ids'] = []
        for option in data['options-for-correct-answers']:
            processed['options'].append(
                self.sentence_to_indices(option['utterance'].lower())
            )
            processed['option_ids'].append(option['candidate-id'])

        for option in data['options-for-next']:
            processed['options'].append(
                self.sentence_to_indices(option['utterance'].lower())
            )
            processed['option_ids'].append(option['candidate-id'])

        processed['n_corrects'] = len(data['options-for-correct-answers'])

        if cat:
            context = []
            utterance_ends = []
            assert len(processed['speaker']) == len(processed['context'])
            for speaker, utterance in zip(processed['speaker'],
                                          processed['context']):
                context.append(
                    self.embeddings.to_index('speaker{}'.format(speaker + 1))
                )
                context += utterance
                utterance_ends.append(len(context) - 1)

            processed['context'] = context
            processed['utterance_ends'] = utterance_ends

            for speaker in [0, 1]:
                context = []
                utterance_ends = []
                for utterance in processed['context{}'.format(speaker + 1)]:
                    context.append(
                        self.embeddings.to_index('speaker{}'.format(speaker + 1))
                    )
                    context += utterance
                    utterance_ends.append(len(context) - 1)
                processed['context{}'.format(speaker + 1)] = context
                processed['utterance_ends{}'.format(speaker + 1)] = utterance_ends

            for speaker in [0, 1]:
                context = []
                utterance_ends = []
                for utterance in processed['context{}_masked'.format(speaker + 1)]:
                    context.append(
                        self.embeddings.to_index('speaker{}'.format(speaker + 1))
                    )
                    context += utterance
                    utterance_ends.append(len(context) - 1)
                processed['utterance_ends{}_masked'.format(speaker + 1)] = utterance_ends
                processed['context{}_masked'.format(speaker + 1)] = context

        if 'profile' in data:
            profile = data['profile']['Courses']
            priors = [
                self.embeddings.to_index(
                    course['offering'].split('-')[0].lower()
                )
                for course in profile['Prior']
            ]
            suggested = [
                self.embeddings.to_index(
                    course['offering'].split('-')[0].lower()
                )
                for course in profile['Suggested']
            ]
            processed['prior'] = [
                1 if w in priors else 0
                for w in processed['context']
            ]
            processed['suggested'] = [
                1 if w in suggested else 0
                for w in processed['context']
            ]
            processed['option_prior'] = [
                [1 if w in priors else 0
                 for w in opt]
                for opt in processed['options']
            ]
            processed['option_suggested'] = [
                [1 if w in suggested else 0
                 for w in opt]
                for opt in processed['options']
            ]
        else:
            processed['prior'] = [
                0 for w in processed['context']
            ]
            processed['suggested'] = [
                0 for w in processed['context']
            ]
            processed['option_prior'] = [
                [0 for w in opt]
                for opt in processed['options']
            ]
            processed['option_suggested'] = [
                [0 for w in opt]
                for opt in processed['options']
            ]

        return processed
