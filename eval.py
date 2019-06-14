from run_bigram_coherence import run_bigram_coherence
from utils.logging_utils import _get_logger
from add_args import add_bigram_args
from datetime import datetime
import config
import numpy as np
import argparse
import gc

def experiments(args):
    runs = 5
    time_str = datetime.now().date().isoformat()
    logname = "[Data@%s]_[Encoder@%s]" % (args.data_name, args.sent_encoder)
    if args.bidirectional:
        logname += "_[Bi]"
    logname += "_%s.log" % time_str
    logger = _get_logger(config.LOG_PATH, logname)
    dis_accs = []
    ins_accs = []
    for i in range(runs):
        dis_acc, ins_acc = run_bigram_coherence(args)
        dis_accs.append(dis_acc[0])
        ins_accs.append(ins_acc[0])
        for _ in range(10):
            gc.collect()

    logger.info("=" * 50)
    for i in range(runs):
        logger.info("Run %d" % (i + 1))
        logger.info("Dis Acc: %.6f" % dis_accs[i])
        logger.info("Ins Acc: %.6f" % ins_accs[i])
    logger.info("=" * 50)
    logger.info("Average Dis Acc: %.6f (%.6f)" % (np.mean(dis_accs), np.std(dis_accs)))
    logger.info("Average Ins Acc: %.6f (%.6f)" % (np.mean(ins_accs), np.std(ins_accs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_bigram_args(parser)
    args = parser.parse_args()

    experiments(args)
