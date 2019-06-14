import os
import logging
import config
from utils.logging_utils import _set_basic_logging
from utils.data_utils import DataSet
from models.infersent_models import InferSent
from models.language_models import LanguageModel
from models.seq2seq_models import Seq2SeqModel
import torch
import numpy as np
import random
import copy
import itertools
import pickle
from tqdm import tqdm
import argparse

def permute_articles(cliques, num_perm):
    permuted_articles = []
    for clique in cliques:
        clique = list(clique)
        old_clique = copy.deepcopy(clique)
        random.shuffle(clique)
        perms = itertools.permutations(clique)
        inner_perm = []
        i = 0
        for perm in perms:
            comparator = [old_sent == sent for old_sent, sent
                          in zip(old_clique, perm)]
            if not np.all(comparator):
                inner_perm.append(list(perm))
                i += 1
            if i >= num_perm:
                break
        permuted_articles.append(inner_perm)
    return permuted_articles

def permute_articles_with_replacement(cliques, num_perm):
    permuted_articles = []
    for clique in cliques:
        clique = list(clique)
        old_clique = copy.deepcopy(clique)
        inner_perm = []
        i = 0
        while i < num_perm:
            random_perm = copy.deepcopy(clique)
            random.shuffle(random_perm)
            comparator = [old_sent == sent for old_sent, sent
                          in zip(old_clique, random_perm)]
            if not np.all(comparator):
                inner_perm.append(random_perm)
                i += 1
            if i >= num_perm:
                break
        permuted_articles.append(inner_perm)
    return permuted_articles

def prep_wsj_lm_data(data_path):
    train_list = ['00', '01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10']

    valid_list = ['11', '12', '13']

    test_list = ['14', '15', '16', '17', '18', '19', '20',
                 '21', '22', '23', '24']

    datasets = [('train', train_list),
                ('valid', valid_list),
                ('test', test_list)]

    for dname, dlist in datasets:
        with open(os.path.join('./', dname+'.txt'), 'w') as wr:
            for dirname in os.listdir(data_path):

                if dirname in dlist:
                    print(dname, dirname)
                    subdirpath = os.path.join(data_path, dirname)

                    for filename in os.listdir(subdirpath):
                        fname = os.path.join(subdirpath, filename)

                        with open(fname) as fr:
                            wr.write("<SOA>"+"\n")
                            wr.write(fr.read().strip()+'\n')
                            wr.write("<EOA>"+"\n")

def load_wsj_file_list(data_path):
    dir_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08',
                '09', '10', '11', '12', '13', '14', '15', '16', '17',
                '18', '19', '20', '21', '22', '23', '24']

    file_list = []
    for dirname in os.listdir(data_path):
        if dirname in dir_list:
            subdirpath = os.path.join(data_path, dirname)
            for filename in os.listdir(subdirpath):
                file_list.append(os.path.join(subdirpath, filename))
    return file_list

def load_wiki_file_list(data_path, dir_list):
    file_list = []
    for dirname in os.listdir(data_path):
        if dirname in dir_list:
            subdirpath = os.path.join(data_path, dirname)
            file_list.append(os.path.join(subdirpath, "extracted_paras.txt"))
    return file_list

def load_file_list(data_name, if_sample):
    if data_name in ["wsj", "wsj_bigram", "wsj_trigram"]:
        if if_sample:
            return load_wsj_file_list(config.SAMPLE_WSJ_DATA_PATH)
        return load_wsj_file_list(config.WSJ_DATA_PATH)
    elif data_name in ["wiki_random", "wiki_bigram_easy"]:
        dir_list = config.WIKI_EASY_TRAIN_LIST + config.WIKI_EASY_TEST_LIST
        if if_sample:
            return load_wiki_file_list(config.SAMPLE_WIKI_DATA_PATH, dir_list)
        return load_wiki_file_list(config.WIKI_EASY_DATA_PATH, dir_list)
    elif (data_name in ["wiki_domain"]) or ("wiki_bigram" in data_name):
        category = data_name[12:]
        if category in config.WIKI_OUT_DOMAIN:
            dir_list = config.WIKI_IN_DOMAIN + [category]
        else:
            dir_list = config.WIKI_IN_DOMAIN
        if if_sample:
            return load_wiki_file_list(config.SAMPLE_WIKI_DATA_PATH, dir_list)
        return load_wiki_file_list(config.WIKI_DATA_PATH, dir_list)
    else:
        raise ValueError("Invalid data name!")

def get_infersent(data_name, on_gpu=True, if_sample=False, return_model=False):
    logging.info("Start parsing...")
    file_list = load_file_list(data_name, if_sample)

    sentences = []
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if (line != '<para_break>') and (line != ''):
                    sentences.append(line)
    logging.info("%d sentences in total." % len(sentences))

    logging.info("Loading infersent models...")
    params = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': 1
    }
    model = InferSent(params)
    model.load_state_dict(torch.load(config.INFERSENT_MODEL))
    model.set_w2v_path(config.WORD_EMBEDDING)
    vocab_size = 10000 if if_sample else 2196017
    model.build_vocab_k_words(K=vocab_size)
    if on_gpu:
        model.cuda()

    logging.info("Encoding sentences...")
    embeddings = model.encode(
        sentences, 128, config.MAX_SENT_LENGTH, tokenize=False, verbose=True)
    logging.info("number of sentences encoded: %d" % len(embeddings))

    assert len(sentences) == len(embeddings), "Lengths don't match!"
    embed_dict = dict(zip(sentences, embeddings))
    np.random.seed(0)
    embed_dict["<SOA>"] = np.random.uniform(size=4096).astype(np.float32)
    embed_dict["<EOA>"] = np.random.uniform(size=4096).astype(np.float32)

    if return_model:
        return embed_dict, model
    else:
        return embed_dict

def get_average_glove(data_name, if_sample=False):
    logging.info("Start parsing...")
    file_list = load_file_list(data_name, if_sample)

    sentences = []
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if (line != '<para_break>') and (line != ''):
                    sentences.append(line)
    logging.info("%d sentences in total." % len(sentences))

    logging.info("Loading glove...")
    word_vec = {}
    with open(config.WORD_EMBEDDING) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec, sep=' ')

    embed_dict = {}
    for s in sentences:
        tokens = s.split()
        embed_dict[s] = np.zeros(300, dtype=np.float32)
        sent_len = 0
        for token in tokens:
            if token in word_vec:
                embed_dict[s] += word_vec[token]
                sent_len += 1
        if sent_len > 0:
            embed_dict[s] = np.true_divide(embed_dict[s], sent_len)
    np.random.seed(0)
    embed_dict["<SOA>"] = np.random.uniform(size=300).astype(np.float32)
    embed_dict["<EOA>"] = np.random.uniform(size=300).astype(np.float32)
    return embed_dict

def get_lm_hidden(data_name, lm_name, corpus):
    logging.info("Start parsing...")
    file_list = load_file_list(data_name, False)

    sentences = []
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if (line != '<para_break>') and (line != ''):
                    sentences.append(line)
    logging.info("%d sentences in total." % len(sentences))

    with open(os.path.join(config.CHECKPOINT_PATH, lm_name + "_forward.pkl"), "rb") as f:
        hparams = pickle.load(f)

    kwargs = {
        "vocab_size": corpus.glove_embed.shape[0],
        "embed_dim": corpus.glove_embed.shape[1],
        "corpus": corpus,
        "hparams": hparams,
    }

    forward_lm = LanguageModel(**kwargs)
    forward_lm.load(os.path.join(config.CHECKPOINT_PATH, lm_name + "_forward.pt"))
    forward_lm = forward_lm.lm
    forward_lm.eval()

    backward_lm = LanguageModel(**kwargs)
    backward_lm.load(os.path.join(config.CHECKPOINT_PATH, lm_name + "_backward.pt"))
    backward_lm = backward_lm.lm
    backward_lm.eval()

    embed_dict = {}
    ini_hidden = forward_lm.init_hidden(1)
    for sent in tqdm(sentences):
        fs = [corpus.vocab[w] for w in ['<eos>'] + sent.split() + ['<eos>']]
        fs = torch.LongTensor(fs).unsqueeze(1)
        fs = fs.to('cuda')
        fout = forward_lm.encode(fs, ini_hidden)
        fout = torch.max(fout, 0)[0].squeeze().data.cpu().numpy().astype(np.float32)

        bs = [corpus.vocab[w] for w in ['<eos>'] + sent.split()[::-1] + ['<eos>']]
        bs = torch.LongTensor(bs).unsqueeze(1)
        bs = bs.to('cuda')
        bout = backward_lm.encode(bs, ini_hidden)
        bout = torch.max(bout, 0)[0].squeeze().data.cpu().numpy().astype(np.float32)

        embed_dict[sent] = np.hstack((fout, bout))
    np.random.seed(0)
    embed_dict["<SOA>"] = np.random.uniform(size=2048).astype(np.float32)
    embed_dict["<EOA>"] = np.random.uniform(size=2048).astype(np.float32)
    return embed_dict

def get_s2s_hidden(data_name, model_name, corpus):
    logging.info("Start parsing...")
    file_list = load_file_list(data_name, False)

    sentences = []
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if (line != '<para_break>') and (line != ''):
                    sentences.append(line)
    logging.info("%d sentences in total." % len(sentences))

    with open(os.path.join(config.CHECKPOINT_PATH, model_name + "_forward.pkl"), "rb") as f:
        hparams = pickle.load(f)

    kwargs = {
        "vocab_size": corpus.glove_embed.shape[0],
        "embed_dim": corpus.glove_embed.shape[1],
        "corpus": corpus,
        "hparams": hparams,
    }

    forward_model = Seq2SeqModel(**kwargs)
    forward_model.load(os.path.join(config.CHECKPOINT_PATH, model_name + "_forward.pt"))
    forward_model = forward_model.model
    forward_model.eval()

    backward_model = Seq2SeqModel(**kwargs)
    backward_model.load(os.path.join(config.CHECKPOINT_PATH, model_name + "_backward.pt"))
    backward_model = backward_model.model
    backward_model.eval()

    embed_dict = {}
    for sent in tqdm(sentences):
        fs = [corpus.vocab[w] for w in sent.split() + ['<eos>']]
        # fs_len = torch.LongTensor([len(fs)])
        # fs_len = fs_len.to('cuda')
        fs = torch.LongTensor(fs).unsqueeze(0)
        fs = fs.to('cuda')
        # fout = forward_model.encoding(fs, fs_len)
        fout = forward_model.encode(fs)
        # fout = fout.squeeze().data.cpu().numpy().astype(np.float32)
        fout = torch.max(fout, 1)[0].squeeze().data.cpu().numpy().astype(np.float32)

        bs = [corpus.vocab[w] for w in sent.split()[::-1] + ['<eos>']]
        # bs_len = torch.LongTensor([len(bs)])
        # bs_len = bs_len.to('cuda')
        bs = torch.LongTensor(bs).unsqueeze(0)
        bs = bs.to('cuda')
        # bout = backward_model.encoding(bs, bs_len)
        bout = backward_model.encode(bs)
        # bout = bout.squeeze().data.cpu().numpy().astype(np.float32)
        bout = torch.max(bout, 1)[0].squeeze().data.cpu().numpy().astype(np.float32)

        embed_dict[sent] = np.hstack((fout, bout))
    np.random.seed(0)
    embed_dict["<SOA>"] = np.random.uniform(size=2048).astype(np.float32)
    embed_dict["<EOA>"] = np.random.uniform(size=2048).astype(np.float32)
    return embed_dict

def save_eval_perm(data_name, if_sample=False, random_seed=config.RANDOM_SEED):
    random.seed(random_seed)

    logging.info("Loading valid and test data...")
    if data_name not in config.DATASET:
        raise ValueError("Invalid data name!")
    dataset = DataSet(config.DATASET[data_name])
    # dataset.random_seed = random_seed
    if if_sample:
        valid_dataset = dataset.load_valid_sample()
    else:
        valid_dataset = dataset.load_valid()
    if if_sample:
        test_dataset = dataset.load_test_sample()
    else:
        test_dataset = dataset.load_test()
    valid_df = valid_dataset.article_df
    test_df = test_dataset.article_df

    logging.info("Generating permuted articles...")

    def permute(x):
        x = np.array(x).squeeze()
        # neg_x_list = permute_articles([x], config.NEG_PERM)[0]
        neg_x_list = permute_articles_with_replacement([x], config.NEG_PERM)[0]
        return "<BREAK>".join(["<PUNC>".join(i) for i in neg_x_list])

    valid_df["neg_list"] = valid_df.sentences.map(permute)
    valid_df["sentences"] = valid_df.sentences.map(lambda x: "<PUNC>".join(x))
    valid_nums = valid_df.neg_list.map(lambda x: len(x.split("<BREAK>"))).sum()
    test_df["neg_list"] = test_df.sentences.map(permute)
    test_df["sentences"] = test_df.sentences.map(lambda x: "<PUNC>".join(x))
    test_nums = test_df.neg_list.map(lambda x: len(x.split("<BREAK>"))).sum()

    logging.info("Number of validation pairs %d" % valid_nums)
    logging.info("Number of test pairs %d" % test_nums)

    logging.info("Saving...")
    dataset.save_valid_perm(valid_df)
    dataset.save_test_perm(test_df)
    logging.info("Finished!")

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='wsj_bigram')


if __name__ == "__main__":
    _set_basic_logging()
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    save_eval_perm(args.data_name, False)
