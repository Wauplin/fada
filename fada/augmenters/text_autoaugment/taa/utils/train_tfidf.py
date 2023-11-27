import re
import argparse
import nlpaug.model.word_stats as nmw
from .raw_data_utils import get_examples
import os
from theconf import Config as C
from datasets import load_dataset 


def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def train_tfidf(dataset, name):
    abspath = C.get()['abspath']
    model_path = '{}/models/tfidf/{}'.format(abspath, name)

    print('Make TF-IDF model directory: %s' % model_path)
    os.makedirs(model_path, exist_ok=True)
    examples = get_examples(dataset, 'text')
    texts = [d.text_a if d.text_b is None else d.text_a + ' ' + d.text_b for d in examples]

    # Tokenize input
    train_x_tokens = [_tokenizer(x) for x in texts]  # List[List[str]]

    # Train TF-IDF models
    print('Start training TF-IDF model. It will take a long time if the training dataset is too large')
    tfidf_model = nmw.TfIdf()
    tfidf_model.train(train_x_tokens)
    tfidf_model.save(model_path)

