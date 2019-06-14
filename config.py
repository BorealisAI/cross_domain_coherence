from utils.data_utils import WSJ_Bigram_Dataset, WIKI_Bigram_Dataset

# ------------------- PATH -------------------
ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
LOG_PATH = "%s/log" % ROOT_PATH
CHECKPOINT_PATH = "%s/checkpoint" % ROOT_PATH

# ------------------- DATA -------------------

INFERSENT_MODEL = "%s/infersent1.pkl" % DATA_PATH
WORD_EMBEDDING = "%s/glove.840B.300d.txt" % DATA_PATH

DATASET = {}

WSJ_DATA_PATH = "%s/parsed_wsj" % DATA_PATH
SAMPLE_WSJ_DATA_PATH = "%s/parsed_wsj" % DATA_PATH
WSJ_VALID_PERM = "%s/valid_perm.tsv" % WSJ_DATA_PATH
WSJ_TEST_PERM = "%s/test_perm.tsv" % WSJ_DATA_PATH

DATASET["wsj_bigram"] = {
    "dataset": WSJ_Bigram_Dataset,
    "data_path": WSJ_DATA_PATH,
    "sample_path": SAMPLE_WSJ_DATA_PATH,
    "valid_perm": WSJ_VALID_PERM,
    "test_perm": WSJ_TEST_PERM,
    "kwargs": {},
}

WIKI_DATA_PATH = DATA_PATH
SAMPLE_WIKI_DATA_PATH = DATA_PATH
WIKI_IN_DOMAIN = ["Artist", "Athlete", "Politician", "Writer", "MilitaryPerson",
                  "OfficeHolder", "Scientist"]
WIKI_OUT_DOMAIN = ["Plant", "CelestialBody", "EducationalInstitution"]

WIKI_EASY_DATA_PATH = "%s/parsed_random" % DATA_PATH
WIKI_EASY_VALID_PERM = "%s/valid_perm.tsv" % WIKI_EASY_DATA_PATH
WIKI_EASY_TEST_PERM = "%s/test_perm.tsv" % WIKI_EASY_DATA_PATH
WIKI_EASY_TRAIN_LIST = ["train"]
WIKI_EASY_TEST_LIST = ["test"]

for i in range(7):
    category = WIKI_IN_DOMAIN[i]
    DATASET["wiki_bigram_%s" % category] = {
        "dataset": WIKI_Bigram_Dataset,
        "data_path": WIKI_DATA_PATH,
        "sample_path": SAMPLE_WIKI_DATA_PATH,
        "valid_perm": "%s/wiki_%s_valid_perm.tsv" % (DATA_PATH, category.lower()),
        "test_perm": "%s/wiki_%s_test_perm.tsv" % (DATA_PATH, category.lower()),
        "kwargs": {
            "train_list": WIKI_IN_DOMAIN[:i] + WIKI_IN_DOMAIN[i + 1:],
            "test_list": [category],
        },
    }

for category in WIKI_OUT_DOMAIN:
    DATASET["wiki_bigram_%s" % category] = {
        "dataset": WIKI_Bigram_Dataset,
        "data_path": WIKI_DATA_PATH,
        "sample_path": SAMPLE_WIKI_DATA_PATH,
        "valid_perm": "%s/wiki_%s_valid_perm.tsv" % (DATA_PATH, category.lower()),
        "test_perm": "%s/wiki_%s_test_perm.tsv" % (DATA_PATH, category.lower()),
        "kwargs": {
            "train_list": WIKI_IN_DOMAIN,
            "test_list": [category],
        },
    }

DATASET["wiki_bigram_easy"] = {
    "dataset": WIKI_Bigram_Dataset,
    "data_path": WIKI_EASY_DATA_PATH,
    "sample_path": SAMPLE_WIKI_DATA_PATH,
    "valid_perm": WIKI_EASY_VALID_PERM,
    "test_perm": WIKI_EASY_TEST_PERM,
    "kwargs": {
        "train_list": WIKI_EASY_TRAIN_LIST,
        "test_list": WIKI_EASY_TEST_LIST,
    },
}

# ------------------- PARAM ------------------

RANDOM_SEED = 2018

MAX_SENT_LENGTH = 40

NEG_PERM = 20
