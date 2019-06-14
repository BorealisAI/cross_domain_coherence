import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from .gan_models import RNN_LM
import numpy as np
import pickle
import time
from tqdm import tqdm

def repackage_hidden(h):
    if isinstance(h, tuple):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()

def batchify(data, bsz, use_cuda):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len])
    target = Variable(source[i + 1: i + 1 + seq_len])
    return data, target

class LanguageModel:
    def __init__(self, vocab_size, embed_dim, corpus, hparams):
        self.vocab_size = vocab_size
        self.hparams = hparams
        self.num_epochs = hparams["num_epochs"]
        self.batch_size = hparams["batch_size"]
        self.bptt = hparams["bptt"]
        self.log_interval = hparams["log_interval"]
        self.save_path = hparams["save_path"]
        lr = hparams["lr"]
        wdecay = hparams["wdecay"]
        self.hparams = hparams
        self.use_cuda = torch.cuda.is_available()

        self.train_data = batchify(corpus.train, self.batch_size, self.use_cuda)
        self.valid_data = batchify(corpus.valid, self.batch_size, self.use_cuda)
        self.lm = RNN_LM(vocab_size, embed_dim, hparams, self.use_cuda)
        if self.use_cuda:
            self.lm.cuda()
        self.lm.set_embed(self._variable(corpus.glove_embed))
        model_parameters = filter(lambda p: p.requires_grad, self.lm.parameters())
        self.optimizer = optim.Adam(model_parameters,
                                    lr=lr, weight_decay=wdecay)
        self.loss_fn = nn.CrossEntropyLoss()

    def _variable(self, data):
        data = np.array(data)
        data = Variable(torch.from_numpy(data))
        return data.cuda() if self.use_cuda else data

    def logging(self, s, print_=True):
        if print_:
            print(s)

    def train(self, epoch):
        self.lm.train()

        total_loss = 0
        hidden = self.lm.init_hidden(self.batch_size)
        i, batch = np.random.choice(self.bptt), 0
        start_time = time.time()

        while i < self.train_data.size(0) - 1 - 1:
            data, target = get_batch(self.train_data, i, self.bptt)
            self.optimizer.zero_grad()

            output, hidden = self.lm(data, hidden)
            hidden = repackage_hidden(hidden)

            loss = self.loss_fn(output.view(-1, self.vocab_size), target.view(-1))
            total_loss += loss.data
            loss.backward()
            self.optimizer.step()

            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / self.log_interval
                elapsed = time.time() - start_time
                self.logging('| epoch {:2d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | '
                             'loss {:.5f} | ppl {:3.3f}'.format(
                                 epoch, batch, len(self.train_data) // self.bptt,
                                 elapsed * 1000 / self.batch_size, cur_loss, np.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

            batch += 1
            i += self.bptt

    def evaluate(self, data_source):
        self.lm.eval()

        total_loss = 0
        hidden = self.lm.init_hidden(self.batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, target = get_batch(data_source, i, self.bptt)

                output, hidden = self.lm(data, hidden)
                hidden = repackage_hidden(hidden)

                loss = self.loss_fn(output.view(-1, self.vocab_size), target.view(-1))
                total_loss += loss.data * len(data)
        return total_loss.item() / len(data_source)

    def fit(self):
        best_valid_loss = np.inf
        best_valid_epoch = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch)
            val_loss = self.evaluate(self.valid_data)
            self.logging('-' * 50)
            self.logging('| end of epoch {:2d} | time {:5.2f}s | valid loss {:.5f} | '
                         'valid ppl {:3.3f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss, np.exp(val_loss)))
            self.logging('-' * 50)
            if val_loss < best_valid_loss:
                self.save(self.save_path)
                best_valid_loss = val_loss
                best_valid_epoch = epoch
            elif epoch - best_valid_epoch > 5:
                break
        return best_valid_loss

    def save(self, path):
        torch.save(self.lm.state_dict(), path + ".pt")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.hparams, f, -1)

    def load(self, path):
        self.lm.load_state_dict(torch.load(path))

class LMCoherence:
    def __init__(self, forward_lm, backward_lm, corpus):
        self.forward_lm = forward_lm
        self.backward_lm = backward_lm
        self.corpus = corpus
        self.use_cuda = torch.cuda.is_available()

        self.loss = nn.CrossEntropyLoss()

    def score_article(self, sentences, reverse=False):
        vocab_size = len(self.corpus.vocab)
        lm = self.backward_lm if reverse else self.forward_lm

        if reverse:
            sentences = sentences[::-1]

        if reverse:
            sentences_inds = [[self.corpus.vocab[w]
                               for w in ['<eos>'] + sent.split()[::-1] + ['<eos>']]
                              for sent in sentences]
        else:
            sentences_inds = [[self.corpus.vocab[w]
                               for w in ['<eos>'] + sent.split() + ['<eos>']]
                              for sent in sentences]

        scores = []
        hidden_f = lm.init_hidden(1)
        for s in sentences_inds:
            s = torch.LongTensor(s).unsqueeze(1)
            if self.use_cuda:
                s = s.to('cuda')
            x = s[:-1]
            y = s[1:].squeeze()

            c_f_outs, hidden_f = lm(x, hidden_f)
            c_loss_f = self.loss(c_f_outs.view(-1, vocab_size), y.view(-1))

            scores.append(- c_loss_f.item())

        return np.mean(scores)

    def evaluate_dis(self, test, df):
        self.forward_lm.eval()
        self.backward_lm.eval()

        correct_pred = 0
        total = 0
        for article in tqdm(test):
            if total % 2000 == 0 and total:
                print(correct_pred / total)
            sentences = df.loc[article[0], "sentences"].split("<PUNC>")
            neg_sentences_list = df.loc[article[0], "neg_list"].split("<BREAK>")
            neg_sentences_list = [s.split('<PUNC>') for s in neg_sentences_list]

            pos_score_f = self.score_article(sentences)
            pos_score_b = self.score_article(sentences, True)
            pos_score = pos_score_f + pos_score_b

            for neg_sentences in neg_sentences_list:
                neg_score_f = self.score_article(neg_sentences)
                neg_score_b = self.score_article(neg_sentences, True)
                neg_score = neg_score_f + neg_score_b
                if pos_score > neg_score:
                    correct_pred += 1
                total += 1
        return correct_pred / total

    def evaluate_ins(self, test, df):
        self.forward_lm.eval()
        self.backward_lm.eval()

        correct_pred = 0.0
        total = 0
        for article in tqdm(test):
            if total % 100 == 0 and total:
                print(correct_pred / total)
            sentences = df.loc[article[0], "sentences"].split("<PUNC>")
            sent_num = len(sentences)

            pos_score_f = self.score_article(sentences)
            pos_score_b = self.score_article(sentences, True)
            pos_score = pos_score_f + pos_score_b

            cnt = 0.0
            for i in range(sent_num):
                tmp = sentences[:i] + sentences[i + 1:]
                flag = True
                for j in range(sent_num):
                    if j == i:
                        continue
                    neg_sentences = tmp[:j] + sentences[i:i + 1] + tmp[j:]
                    neg_score_f = self.score_article(neg_sentences)
                    neg_score_b = self.score_article(neg_sentences, True)
                    neg_score = neg_score_f + neg_score_b
                    if pos_score < neg_score:
                        flag = False
                if flag:
                    cnt += 1.0
            correct_pred += cnt / sent_num
            total += 1
        return correct_pred / total
