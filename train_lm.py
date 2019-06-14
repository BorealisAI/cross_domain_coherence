from utils.lm_utils import Corpus
from utils.data_utils import DataSet
from models.language_models import LanguageModel
import config
import argparse

def train_lm(args):
    if args.data_name not in config.DATASET:
        raise ValueError("Invalid data name!")
    dataset = DataSet(config.DATASET[args.data_name])
    train_dataset = dataset.load_train()
    test_dataset = dataset.load_test()
    corpus = Corpus(train_dataset.file_list, test_dataset.file_list, args.reverse)
    suffix = "backward" if args.reverse else "forward"

    kwargs = {
        "vocab_size": corpus.glove_embed.shape[0],
        "embed_dim": corpus.glove_embed.shape[1],
        "corpus": corpus,
        "hparams": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "cell_type": args.cell_type,
            "tie_embed": args.tie_embed,
            "rnn_dropout": args.rnn_dropout,
            "hidden_dropout": args.hidden_dropout,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "bptt": args.bptt,
            "log_interval": args.log_interval,
            "save_path": args.save_path + '_' + args.data_name + '_' + suffix,
            "lr": args.lr,
            "wdecay": args.wdecay,
        }
    }

    lm = LanguageModel(**kwargs)
    best_valid_loss = lm.fit()
    print("Best Valid Loss:", best_valid_loss)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default="wsj_bigram",
                        help='data name')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='log interval for training')
    parser.add_argument('--save_path', type=str, default='checkpoint/lm',
                        help='save path')
    parser.add_argument('--reverse', default=False, action='store_true',
                        help='reverse the text')

    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--cell_type', type=str, default='lstm',
                        help='RNN cell type (i.e. rnn, gru or lstm)')
    parser.add_argument('--tie_embed', default=False, action='store_true',
                        help='Tie embedding and softmax weights')
    parser.add_argument('--rnn_dropout', type=float, default=0.5,
                        help='RNN dropout')
    parser.add_argument('--hidden_dropout', type=float, default=0.5,
                        help='hidden dropout')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0.0,
                        help='weight decay')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    train_lm(args)
