from models.language_models import LanguageModel, LMCoherence
from utils.lm_utils import Corpus
from utils.data_utils import DataSet
from utils.logging_utils import _set_basic_logging
import logging
import config
from torch.utils.data import DataLoader
import os
import argparse
import pickle


def run_lm_coherence(args):
    logging.info("Loading data...")
    if args.data_name not in config.DATASET:
        raise ValueError("Invalid data name!")

    dataset = DataSet(config.DATASET[args.data_name])
    train_dataset = dataset.load_train()
    test_df = dataset.load_test_perm()
    test_dataset = dataset.load_test()
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    corpus = Corpus(train_dataset.file_list, test_dataset.file_list)

    # dataset = DataSet(config.DATASET["wsj_bigram"])
    # test_df = dataset.load_test_perm()
    # test_dataset = dataset.load_test()
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    with open(os.path.join(config.CHECKPOINT_PATH, args.lm_name + "_forward.pkl"), "rb") as f:
        hparams = pickle.load(f)

    kwargs = {
        "vocab_size": corpus.glove_embed.shape[0],
        "embed_dim": corpus.glove_embed.shape[1],
        "corpus": corpus,
        "hparams": hparams,
    }

    forward_model = LanguageModel(**kwargs)
    forward_model.load(os.path.join(config.CHECKPOINT_PATH, args.lm_name + "_forward.pt"))
    backward_model = LanguageModel(**kwargs)
    backward_model.load(os.path.join(config.CHECKPOINT_PATH, args.lm_name + "_backward.pt"))

    logging.info("Results for discrimination:")
    model = LMCoherence(forward_model.lm, backward_model.lm, corpus)
    dis_acc = model.evaluate_dis(test_dataloader, test_df)
    logging.info("Disc Accuracy: {}".format(dis_acc))

    logging.info("Results for insertion:")
    ins_acc = model.evaluate_ins(test_dataloader, test_df)
    logging.info("Disc Accuracy: {}".format(ins_acc))

    return dis_acc, ins_acc

def add_args(parser):
    parser.add_argument('--data_name', type=str, default="wiki_bigram_easy",
                        help='data name')
    parser.add_argument('--lm_name', type=str, default="lm_wiki_bigram_easy",
                        help='languange model name')


if __name__ == "__main__":
    _set_basic_logging()

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    run_lm_coherence(args)
