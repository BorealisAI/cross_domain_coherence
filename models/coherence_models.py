import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from .gan_models import MLP_Discriminator
import numpy as np
import pickle
from datetime import datetime
from utils.np_utils import generate_random_pmatrices
from tqdm import tqdm
import utils.lm_utils as utils


class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, p_scores, n_scores, weights=None):
        scores = self.margin + p_scores - n_scores
        scores = scores.clamp(min=0)
        if weights is not None:
            scores = weights * scores
        return scores.mean()

class BigramCoherence:
    def __init__(self, embed_dim, sent_encoder, hparams):
        self.embed_dim = embed_dim
        self.sent_encoder = sent_encoder
        self.num_epochs = hparams["num_epochs"]
        margin = hparams["margin"]
        lr = hparams["lr"]
        l2_reg_lambda = hparams["l2_reg_lambda"]
        self.task = hparams["task"]
        self.hparams = hparams

        self.use_cuda = torch.cuda.is_available()
        self.use_pretrained = isinstance(self.sent_encoder, dict)
        self.discriminator = MLP_Discriminator(
            embed_dim, hparams, self.use_cuda)
        model_parameters = list(self.discriminator.parameters())
        if not self.use_pretrained:
            model_parameters += list(self.sent_encoder.parameters())
        self.optimizer = optim.Adam(model_parameters,
                                    lr=lr, weight_decay=l2_reg_lambda)

        self.loss_name = hparams['loss']
        if hparams['loss'] == 'margin':
            self.loss_fn = MarginRankingLoss(margin)
        elif hparams['loss'] == 'log':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif hparams['loss'] == 'margin+log':
            self.loss_fn = [MarginRankingLoss(margin), nn.BCEWithLogitsLoss()]
        else:
            raise NotImplementedError()

        if self.use_cuda:
            self.discriminator.cuda()
            if not self.use_pretrained:
                self.sent_encoder.cuda()
        self.best_discriminator = self.discriminator.state_dict()
        self.intervals = [0, 10, 20, np.inf]

    def init(self):
        def init_weights(model):
            if type(model) in [nn.Linear]:
                nn.init.xavier_normal_(model.weight.data)

        self.discriminator.apply(init_weights)

    def _variable(self, data):
        data = np.array(data)
        data = Variable(torch.from_numpy(data))
        return data.cuda() if self.use_cuda else data

    def encode(self, sentences):
        if self.use_pretrained:
            sentences = np.array(list(map(self.sent_encoder.get, sentences)))
            sentences = self._variable(sentences)
            return sentences
        sentences, lengths, idx_sort = self.sent_encoder.prepare_samples(
            sentences, -1, 40, False, False)
        with torch.autograd.no_grad():
            batch = Variable(self.sent_encoder.get_batch(sentences))
        if self.use_cuda:
            batch = batch.cuda()
        batch = self.sent_encoder.forward((batch, lengths))
        return batch

    def fit(self, train, valid=None, df=None):
        best_valid_acc = 0
        best_valid_epoch = 0
        step = 0
        for epoch in range(1, self.num_epochs + 1):
            for sentences in train:
                step += 1
                self.discriminator.zero_grad()

                sent1 = []
                pos_sent2 = []
                neg_sent2 = []
                slens = []
                for s in sentences:
                    s1, ps2, ns2, slen = s.split('<BREAK>')
                    sent1.append(s1)
                    pos_sent2.append(ps2)
                    neg_sent2.append(ns2)
                    slens.append(int(slen))
                sent1 = self.encode(sent1)
                slens = np.array(slens, dtype=np.float32)
                weights = 1. / slens / (slens - 1)
                weights /= np.mean(weights)
                weights = self._variable(weights)

                pos_sent2 = self.encode(pos_sent2)
                pos_scores = self.discriminator(sent1, pos_sent2)

                neg_sent2 = self.encode(neg_sent2)
                neg_scores = self.discriminator(sent1, neg_sent2)

                if self.loss_name == 'margin':
                    loss = self.loss_fn(pos_scores, neg_scores)

                elif self.loss_name == 'log':
                    loss = self.loss_fn(-pos_scores,
                                        torch.ones_like(pos_scores))
                    loss += self.loss_fn(-neg_scores,
                                         torch.zeros_like(neg_scores))
                elif self.loss_name == 'margin+log':
                    loss = self.loss_fn[0](pos_scores, neg_scores)
                    loss += .1 * \
                        self.loss_fn[1](-pos_scores,
                                        torch.ones_like(pos_scores))
                    loss += .1 * \
                        self.loss_fn[1](-neg_scores,
                                        torch.zeros_like(neg_scores))
                else:
                    raise NotImplementedError()

                if step % 100 == 0:
                    time_str = datetime.now().isoformat()
                    print("{}: step {}, loss {:g}".format(
                        time_str, step, loss.item()))

                loss.backward()
                self.optimizer.step()

            if valid is not None:
                print("\nValidation:")
                print("previous best epoch {}, acc {:g}".format(
                    best_valid_epoch, best_valid_acc))
                acc, _ = self.evaluate(valid, df, self.task)
                print("epoch {} acc {:g}".format(epoch, acc))
                print("")
                if acc > best_valid_acc:
                    best_valid_acc = acc
                    best_valid_epoch = epoch
                    self.best_discriminator = self.discriminator.state_dict()
                if epoch - best_valid_epoch > 3:
                    break
        return best_valid_epoch, best_valid_acc

    def evaluate(self, test, df, task="discrimination"):
        if task == "discrimination":
            return self.evaluate_dis(test, df)
        elif task == "insertion":
            return self.evaluate_ins(test, df)
        else:
            raise ValueError("Invalid task name!")

    def score_article(self, article, reverse=False):
        if reverse:
            article = article[::-1]

        first_sentences = self.encode(article[:-1])
        second_sentences = self.encode(article[1:])
        y = self.discriminator(first_sentences, second_sentences)
        local_y = y.mean().data.cpu().numpy()

        return local_y

    def evaluate_dis(self, test, df, debug=False):
        correct_pred = [0, 0, 0]
        total_samples = [0, 0, 0]

        if debug:
            all_pos_scores = []
            all_neg_scores = []

        self.discriminator.train(False)
        for article in test:
            sentences = df.loc[article[0], "sentences"].split("<PUNC>")
            sent_num = len(sentences)
            sentences = ["<SOA>"] + sentences + ["<EOA>"]
            neg_sentences_list = df.loc[article[0],
                                        "neg_list"].split("<BREAK>")
            neg_sentences_list = [s.split("<PUNC>")
                                  for s in neg_sentences_list]

            pos_sent1 = sentences[:-1]
            pos_sent1 = self.encode(pos_sent1)
            pos_sent2 = sentences[1:]
            pos_sent2 = self.encode(pos_sent2)
            pos_scores = self.discriminator(pos_sent1, pos_sent2)
            # import ipdb
            # ipdb.set_trace()
            mean_pos_score = pos_scores.mean().data.cpu().numpy()

            if debug:
                all_pos_scores.append(pos_scores.data.cpu().numpy().squeeze())

            mean_neg_scores = []
            for neg_sentences in neg_sentences_list:
                neg_sentences = ["<SOA>"] + neg_sentences + ["<EOA>"]
                neg_sent1 = neg_sentences[:-1]
                neg_sent1 = self.encode(neg_sent1)
                neg_sent2 = neg_sentences[1:]
                neg_sent2 = self.encode(neg_sent2)
                neg_scores = self.discriminator(neg_sent1, neg_sent2)
                mean_neg_score = neg_scores.mean().data.cpu().numpy()
                mean_neg_scores.append(mean_neg_score)

                if debug:
                    all_neg_scores.append(
                        neg_scores.data.cpu().numpy().squeeze())

            for mean_neg_score in mean_neg_scores:
                if mean_pos_score < mean_neg_score:
                    for i in range(3):
                        lower_bound = self.intervals[i]
                        upper_bound = self.intervals[i + 1]
                        if (sent_num > lower_bound) and (sent_num <= upper_bound):
                            correct_pred[i] += 1
                for i in range(3):
                    lower_bound = self.intervals[i]
                    upper_bound = self.intervals[i + 1]
                    if (sent_num > lower_bound) and (sent_num <= upper_bound):
                        total_samples[i] += 1
        self.discriminator.train(True)
        accs = np.true_divide(correct_pred, total_samples)
        acc = np.true_divide(np.sum(correct_pred), np.sum(total_samples))

        if debug:
            all_pos_scores = np.concatenate(all_pos_scores)
            all_neg_scores = np.concatenate(all_neg_scores)

            import pandas as pd

            print('pos score stats')
            print(pd.DataFrame(all_pos_scores).describe())

            print('neg score stats')
            print(pd.DataFrame(all_neg_scores).describe())

        return acc, accs

    def evaluate_ins(self, test, df):
        correct_pred = [0, 0, 0]
        total_samples = [0, 0, 0]
        self.discriminator.train(False)
        for article in tqdm(test, disable=False):
            sentences = df.loc[article[0], "sentences"].split("<PUNC>")
            sent_num = len(sentences)
            sentences = ["<SOA>"] + sentences + ["<EOA>"]

            pos_sent1 = sentences[:-1]
            pos_sent1 = self.encode(pos_sent1)
            pos_sent2 = sentences[1:]
            pos_sent2 = self.encode(pos_sent2)
            pos_scores = self.discriminator(pos_sent1, pos_sent2)
            mean_pos_score = pos_scores.mean().data.cpu().numpy()

            cnt = 0.0
            for i in range(1, sent_num + 1):
                tmp = sentences[:i] + sentences[i + 1:]
                flag = True
                for j in range(1, sent_num + 1):
                    if j == i:
                        continue
                    neg_sentences = tmp[:j] + sentences[i:i + 1] + tmp[j:]
                    neg_sent1 = neg_sentences[:-1]
                    neg_sent1 = self.encode(neg_sent1)
                    neg_sent2 = neg_sentences[1:]
                    neg_sent2 = self.encode(neg_sent2)
                    neg_scores = self.discriminator(neg_sent1, neg_sent2)
                    mean_neg_score = neg_scores.mean().data.cpu().numpy()
                    if mean_pos_score > mean_neg_score:
                        flag = False
                if flag:
                    cnt += 1.0
            for i in range(3):
                lower_bound = self.intervals[i]
                upper_bound = self.intervals[i + 1]
                if (sent_num > lower_bound) and (sent_num <= upper_bound):
                    correct_pred[i] += cnt / sent_num
            for i in range(3):
                lower_bound = self.intervals[i]
                upper_bound = self.intervals[i + 1]
                if (sent_num > lower_bound) and (sent_num <= upper_bound):
                    total_samples[i] += 1
        self.discriminator.train(True)
        accs = np.true_divide(correct_pred, total_samples)
        acc = np.true_divide(np.sum(correct_pred), np.sum(total_samples))
        return acc, accs

    def save(self, path):
        torch.save(self.best_discriminator, path + ".pt")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.hparams, f, -1)

    def load(self, path):
        self.discriminator.load_state_dict(torch.load(path))

    def load_best_state(self):
        self.discriminator.load_state_dict(self.best_discriminator)
