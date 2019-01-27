def collect_words(self, data_path):
    logging.info('Loading train data...')
    train_path = os.path.join(data_path, 'ubuntu_train_subtask1.json')
    with open(train_path) as f:
        train = json.load(f)

    logging.info('Loading valid data...')
    valid_path = os.path.join(data_path, 'ubuntu_dev_subtask1.json')
    with open(valid_path) as f:
        valid = json.load(f)

    self.tokenize_data(train)
    self.tokenized_train = train
    self.tokenize_data(valid)
    self.tokenized_valid = valid

    words = set()
    data = train + valid
    for sample in data:
        utterances = (
            [message['utterance']
             for message in sample['messages-so-far']]
            + [option['utterance']
               for option in sample['options-for-correct-answers']]
            + [option['utterance']
               for option in sample['options-for-next']]
        )

        for utterance in utterances:
            for word in utterance:
                words.add(word)

    return words


def tokenize_data(self, data):
    for sample in tqdm(data):
        for i, message in enumerate(sample['messages-so-far']):
            sample['messages-so-far'][i]['utterance'] = \
                self.tokenize(message['utterance'])
        for i, message in enumerate(sample['options-for-correct-answers']):
            sample['options-for-correct-answers'][i]['utterance'] = \
                self.tokenize(message['utterance'])
        for i, message in enumerate(sample['options-for-next']):
            sample['options-for-next'][i]['utterance'] = \
                self.tokenize(message['utterance'])

