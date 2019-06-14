# Cross-Domain Coherence Modeling 

A Cross-Domain Transferable Neural Coherence Model

Paper published in ACL 2019: (arxiv.org/abs/1905.11912)[https://arxiv.org/abs/1905.11912]

This implementation is based on PyTorch 0.4.1.

### Dataset

To download the dataset:

```
python prepare_data.py
```

which includes WikiCoherence dataset we construct, 300-dim GloVe embeddings and pre-trained Infersent model.

For WikiCoherence, it contains:

- 7 categories under **Person**
    - Artist
    - Athlete
    - Politician
    - Writer
    - MilitaryPerson
    - OfficeHolder
    - Scientist
- 3 categories from different irrelevant domains:
    - Plant
    - EducationalInstitution
    - CelestialBody
- parsed\_wsj: original split for Wall Street Journal (WSJ)
- parsed\_random: randomly split all paragraphs of the seven categories under **Person** into training part and testing part

Check `config.py` for the data\_name for each setting.

### Preprocessing

Premute the original documents or paragraphs to obtain the negative samples for evaluation:

```
python preprocess.py --data_name <data_name>
```

### LM Pre-training

Train the LM with the following command:

```
python train_lm.py --data_name <data_name>
```

The pre-trained models will be saved in `./checkpoint`.

### Training and Evaluation

To evaluate our proposed models:

```
python run_bigram_coherence.py --data_name <data_name> --sent_encoder <sent_encoder> [--bidirectional]
```

where `sent_encoder` can be average\_glove, infersent or lm\_hidden.

```
python eval.py --data_name <data_name> --sent_encoder <sent_encoder> [--bidirectional]
```

Run the above script will run the experiment multiple times and report the mean and std statistics.
The log will be saved in `./log`.

### Cite

If you found this codebase or our work useful, please cite:

```
@InProceddings{xu2019cross,
    author = {Xu, Peng and Saghir, Hamidreza and Kang, Jin Sung and Long, Teng and Bose, Avishek Joey and Cao, Yanshuai and Cheung, Jackie Chi Kit},
    title = {A Cross-Domain Transferable Neural Coherence Model}
    booktitle = {The 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)},
    month = {July},
    year = {2019},
    publisher = {ACL}
}
```
