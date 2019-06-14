from models.coherence_models import BigramCoherence
from preprocess import get_infersent, get_average_glove, save_eval_perm, get_lm_hidden
from preprocess import get_s2s_hidden
from utils.data_utils import DataSet
from utils.lm_utils import Corpus, SentCorpus
from utils.logging_utils import _set_basic_logging
import logging
import config
from torch.utils.data import DataLoader
import os
import argparse
from add_args import add_bigram_args
import torch


def run_bigram_coherence(args):
    logging.info("Loading data...")
    if args.data_name not in config.DATASET:
        raise ValueError("Invalid data name!")
    dataset = DataSet(config.DATASET[args.data_name])
    # dataset.random_seed = args.random_seed
    if not os.path.isfile(dataset.test_perm):
        save_eval_perm(args.data_name, random_seed=args.random_seed)

    train_dataset = dataset.load_train(args.portion)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
    valid_dataset = dataset.load_valid(args.portion)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    valid_df = dataset.load_valid_perm()
    test_dataset = dataset.load_test(args.portion)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    test_df = dataset.load_test_perm()

    logging.info("Loading sent embedding...")
    if args.sent_encoder == "infersent":
        sent_embedding = get_infersent(args.data_name, if_sample=args.test)
        embed_dim = 4096
    elif args.sent_encoder == "average_glove":
        sent_embedding = get_average_glove(args.data_name, if_sample=args.test)
        embed_dim = 300
    elif args.sent_encoder == "lm_hidden":
        corpus = Corpus(train_dataset.file_list, test_dataset.file_list)
        sent_embedding = get_lm_hidden(args.data_name, "lm_" + args.data_name, corpus)
        embed_dim = 2048
    elif args.sent_encoder == "s2s_hidden":
        corpus = SentCorpus(train_dataset.file_list, test_dataset.file_list)
        sent_embedding = get_s2s_hidden(args.data_name, "s2s_" + args.data_name, corpus)
        embed_dim = 2048
    else:
        raise ValueError("Invalid sent encoder name!")

    logging.info("Training BigramCoherence model...")
    kwargs = {
        "embed_dim": embed_dim,
        "sent_encoder": sent_embedding,
        "hparams": {
            "loss": args.loss,
            "input_dropout": args.input_dropout,
            "hidden_state": args.hidden_state,
            "hidden_layers": args.hidden_layers,
            "hidden_dropout": args.hidden_dropout,
            "num_epochs": args.num_epochs,
            "margin": args.margin,
            "lr": args.lr,
            "l2_reg_lambda": args.l2_reg_lambda,
            "use_bn": args.use_bn,
            "task": "discrimination",
            "bidirectional": args.bidirectional,
        }
    }

    model = BigramCoherence(**kwargs)
    model.init()
    best_step, valid_acc = model.fit(train_dataloader, valid_dataloader, valid_df)
    if args.save:
        model_path = os.path.join(config.CHECKPOINT_PATH, "%s-%.4f" % (args.data_name, valid_acc))
        # model.save(model_path)
        torch.save(model, model_path + '.pth')
    model.load_best_state()

    # dataset = DataSet(config.DATASET["wsj_bigram"])
    # test_dataset = dataset.load_test()
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # test_df = dataset.load_test_perm()
    # if args.sent_encoder == "infersent":
    #    model.sent_encoder = get_infersent("wsj_bigram", if_sample=args.test)
    # elif args.sent_encoder == "average_glove":
    #    model.sent_encoder = get_average_glove("wsj_bigram", if_sample=args.test)
    # else:
    #    model.sent_encoder = get_lm_hidden("wsj_bigram", "lm_" + args.data_name, corpus)

    logging.info("Results for discrimination:")
    dis_acc = model.evaluate_dis(test_dataloader, test_df)
    print("Test Acc:", dis_acc)
    logging.info("Disc Accuracy: {}".format(dis_acc[0]))

    logging.info("Results for insertion:")
    ins_acc = model.evaluate_ins(test_dataloader, test_df)
    print("Test Acc:", ins_acc)
    logging.info("Insert Accuracy: {}".format(ins_acc[0]))

    return dis_acc, ins_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_bigram_args(parser)
    args = parser.parse_args()

    _set_basic_logging()
    run_bigram_coherence(args)
