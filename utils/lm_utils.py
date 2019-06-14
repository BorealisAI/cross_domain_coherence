import torch
from collections import Counter
import config
import numpy as np

class Vocabulary(object):
    OOV = 0
    EOS = 1

    def __init__(self):
        self.word2idx = {"<oov>": 0, "<eos>": 1}
        self.idx2word = ["<oov>", "<eos>"]
        self.size = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.size += 1

    def __getitem__(self, word):
        return self.word2idx.get(word, 0)

    def to_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return self.size

class Corpus(object):
    def __init__(self, train_list, test_list, reverse=False):
        self.vocab = Vocabulary()

        corpus_vocab = self.create_vocab(train_list + test_list)
        embedding_file = config.WORD_EMBEDDING

        words = []
        glove_embed = []
        with open(embedding_file) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in corpus_vocab:
                    self.vocab.add_word(word)
                    glove_embed.append(np.fromstring(
                        vec, sep=' ', dtype=np.float32))
                    words.append(word)

        self.glove_embed = np.vstack(glove_embed)
        _mu = self.glove_embed.mean()
        _std = self.glove_embed.std()
        self.glove_embed = np.vstack([np.random.randn(
            2, self.glove_embed.shape[1]) * _std + _mu, self.glove_embed]).astype(np.float32)
        print(self.glove_embed.shape)

        train = self.tokenize(train_list, reverse)
        train_len = int(train.shape[0] * 0.9)

        self.train = train[:train_len]
        self.valid = train[train_len:]
        self.test = self.tokenize(test_list, reverse)

    def create_vocab(self, file_list, top=100000):
        counter = Counter()
        for article in file_list:
            with open(article) as f:
                for line in f:
                    line = line.strip()
                    if line == '<para_break>':
                        continue
                    for word in line.split():
                        counter[word] += 1
        counter = sorted(counter, key=counter.get, reverse=True)
        counter = counter[:top]
        return set(counter)

    def tokenize(self, file_list, reverse=False):
        words = []
        for article in file_list:
            with open(article) as f:
                for line in f:
                    line = line.strip()
                    if (line == '<para_break>') or (line == ''):
                        continue
                    for word in line.split() + ['<eos>']:
                        words.append(word)

        idxs = [self.vocab[w] for w in words]
        if reverse:
            idxs = idxs[::-1]
        idxs = torch.LongTensor(idxs)
        return idxs

class SentCorpus(object):
    def __init__(self, train_list, test_list, reverse=False, max_len=40, shuffle=True):
        self.vocab = Vocabulary()
        self.reverse = reverse
        self.shuffle = shuffle
        
        corpus_vocab = self.create_vocab(train_list + test_list)
        embedding_file = config.WORD_EMBEDDING

        words = []
        glove_embed = []
        with open(embedding_file) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in corpus_vocab:
                    self.vocab.add_word(word)
                    glove_embed.append(np.fromstring(
                        vec, sep=' ', dtype=np.float32))
                    words.append(word)

        self.glove_embed = np.vstack(glove_embed)
        _mu = self.glove_embed.mean()
        _std = self.glove_embed.std()
        self.glove_embed = np.vstack([np.random.randn(
            2, self.glove_embed.shape[1]) * _std + _mu, self.glove_embed]).astype(np.float32)
        print(self.glove_embed.shape)

        data = self.get_data(train_list)

        train_len = int(len(data) * 0.9)

        self.train = data[:train_len]
        self.valid = data[train_len:]
        self.max_len = max_len

    def create_vocab(self, file_list, top=100000):
        counter = Counter()
        for article in file_list:
            with open(article) as f:
                for line in f:
                    line = line.strip()
                    if line == '<para_break>':
                        continue
                    for word in line.split():
                        counter[word] += 1
        counter = sorted(counter, key=counter.get, reverse=True)
        counter = counter[:top]
        return set(counter)

    def get_data(self, file_list):
        source = []
        target = []
        for article in file_list:
            sentences = []
            with open(article) as f:
                for line in f:
                    line = line.strip()
                    if (line == '<para_break>') or (line == ''):
                        continue
                    sentences.append(line)
            source.extend(sentences[:-1])
            target.extend(sentences[1:])
        if self.reverse:
            return list(zip(target, source))
        else:
            return list(zip(source, target))

    def tokenize(self, sent, reverse=False):
        words = sent.split()
        if reverse or self.reverse:
            words = words[::-1]
        indices = [self.vocab[w] for w in words][:self.max_len - 1]
        indices += [self.vocab.EOS] * (self.max_len - len(indices))
        return indices

    def fetch_train_batches(self, batch_size):
        data_size = len(self.train)
        nbatch = data_size // batch_size

        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
        else:
            shuffle_indices = np.arange(data_size)
        
        for i in range(nbatch):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            source, source_len = [], []
            target, target_len = [], []
            for j in shuffle_indices[start_index:end_index]:
                s, t = self.train[j]
                source_len.append(min(len(s.split()) + 1, self.max_len))
                source.append(self.tokenize(s))
                target_len.append(min(len(t.split()) + 1, self.max_len))
                target.append(self.tokenize(t))
            yield source, source_len, target, target_len

    def fetch_valid_batches(self, batch_size):
        data_size = len(self.valid)
        nbatch = data_size // batch_size
        for i in range(nbatch):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            source, source_len = [], []
            target, target_len = [], []
            for j in range(start_index, end_index):
                s, t = self.valid[j]
                source_len.append(min(len(s.split()) + 1, self.max_len))
                source.append(self.tokenize(s))
                target_len.append(min(len(t.split()) + 1, self.max_len))
                target.append(self.tokenize(t))
            yield source, source_len, target, target_len
