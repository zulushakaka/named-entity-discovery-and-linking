import sys
import os
import heapq
from itertools import count
import random

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid

import numpy as np
import torch
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer
from neuronlp2.models import BiRecurrentConvCRF, Embedding, ChainCRF
from neuronlp2 import utils

from dictionary import is_url
from ner import evaluate


def preprocess():
    root = '/media/xianyang/OS/CMU/opera/data/txt'
    target = open('temp/unannotated.conll', 'w')
    for fname in os.listdir(root):
        if fname.endswith('.txt'):
            with open(os.path.join(root, fname), 'r') as f:
                for line in f:
                    if len(line) > 1:
                        for wid, word in enumerate(line.strip().split()):
                            target.write('{0} {1} -- -- O\n'.format(wid+1, word))
                        target.write('\n')
    target.close()

def sample():
    network = torch.load('temp/ner_active.pt')
    word_alphabet, char_alphabet, pos_alphabet, \
        chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("active_alphabet/", None)

    unannotated_data = conll03_data.read_data_to_variable('temp/unannotated.conll', word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, 
        use_gpu=False, volatile=True)

    annotated = set()
    with open('temp/annotated.conll', 'r') as f:
        sent_buffer = []
        for line in f:
            if len(line) > 1:
                _, word, _, _, _ = line.strip().split()
                sent_buffer.append(word)
            else:
                annotated.add(' '.join(sent_buffer))
                sent_buffer = []
    print('total annotated data: {}'.format(len(annotated)))

    uncertain = []
    max_sents = 100
    max_words = 500

    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    writer.start('temp/output.txt')
    network.eval()
    tiebreaker = count()
    for batch in conll03_data.iterate_batch_variable(unannotated_data, 32):
        word, char, pos, chunk, labels, masks, lengths, raws = batch
        preds, _, confidence = network.decode(word, char, target=labels, mask=masks,
                                          leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
        writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                             preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
        for _ in range(confidence.size()[0]):
            heapq.heappush(uncertain, (confidence[_].numpy()[0] / lengths[_], tiebreaker.next(), word[_].data.numpy(), raws[_]))
    writer.close()

    cost_sents = 0
    cost_words = 0
    with open('temp/query.conll', 'w') as q:
        while cost_sents < max_sents and cost_words < max_words and uncertain:
            sample = heapq.heappop(uncertain)
            if len(sample[3]) <= 5:
                continue
            # print(sample[0])
            # print([word_alphabet.get_instance(wid) for wid in sample[2]])
            print(sample[3])
            to_write = []
            for word in sample[3]:
                if is_url(word):
                    word = '<_URL>'
                to_write.append(word.encode('ascii', 'ignore'))
            if ' '.join(to_write) in annotated:
                continue
            for wn, word in enumerate(to_write):
                q.write('{0} {1} -- -- O\n'.format(wn+1, word))
            q.write('\n')
            cost_sents += 1
            cost_words += len(sample[3])
    
def retrain(train_path, dev_path):
    network = torch.load('temp/ner_tuned.pt')
    word_alphabet, char_alphabet, pos_alphabet, \
        chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("ner_alphabet/", None)
    
    num_new_word = 0
    with open(train_path, 'r') as f:
        sents = []
        sent_buffer = []
        for line in f:
            if len(line) <= 1:
                sents.append(sent_buffer)
                sent_buffer = []
            else:
                id, word, _, _, ner = line.strip().split()
                if word_alphabet.get_index(word) == 0:
                    word_alphabet.add(word)
                    num_new_word += 1
                sent_buffer.append((word_alphabet.get_index(word), ner_alphabet.get_index(ner)))
    print('{} new words.'.format(num_new_word))
    init_embed = network.word_embedd.weight.data
    embedd_dim = init_embed.shape[1]
    init_embed = np.concatenate((init_embed, np.zeros((num_new_word, embedd_dim))), axis=0)
    network.word_embedd = Embedding(word_alphabet.size(), embedd_dim, torch.from_numpy(init_embed))

    target_train_data = conll03_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=False)

    num_epoch = 50
    batch_size = 20
    num_data = sum(target_train_data[1])
    num_batches = num_data / batch_size + 1
    unk_replace = 0.0
    optim = SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=True)

    for epoch in range(num_epoch):
        train_err = 0.
        train_total = 0.
        start_time = time.time()
        num_back = 0
        network.train()

        for batch in range(1, num_batches + 1):
            word, char, _, _, labels, masks, lengths = conll03_data.get_batch_variable(target_train_data, batch_size,
                                                                                       unk_replace=unk_replace)

            optim.zero_grad()
            loss = network.loss(word, char, labels, mask=masks)
            loss.backward()
            optim.step()

            num_inst = word.size(0)
            train_err += loss.data[0] * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

    torch.save(network, 'temp/ner_active.pt')
    alphabet_directory = 'active_alphabet/'
    word_alphabet.save(alphabet_directory)
    char_alphabet.save(alphabet_directory)
    pos_alphabet.save(alphabet_directory)
    chunk_alphabet.save(alphabet_directory)
    ner_alphabet.save(alphabet_directory)

    target_dev_data = conll03_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=False)
    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    os.system('rm output.txt')
    writer.start('output.txt')
    network.eval()
    for batch in conll03_data.iterate_batch_variable(target_dev_data, batch_size):
        word, char, pos, chunk, labels, masks, lengths, _ = batch
        preds, _, _ = network.decode(word, char, target=labels, mask=masks,
                                          leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
        writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                             preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
    writer.close()

    acc, precision, recall, f1 = evaluate('output.txt')
    print(acc, precision, recall, f1)
    return acc, precision, recall, f1

def cross_validate():
    data = []
    with open('temp/annotated.conll', 'r') as f:
        sent_buffer = []
        for line in f:
            if len(line) > 1:
                _, word, _, _, ner = line.strip().split()
                sent_buffer.append((word, ner))
            else:
                data.append(sent_buffer)
                sent_buffer = []
    data_size = len(data)
    print('total data: {}'.format(data_size))
    random.shuffle(data)

    K = 10
    split = data_size / K
    score = []
    for round in range(K):
        print('round {}'.format(round))
        dev = data[split * round : split * (round + 1)]
        train = []
        for i in range(K):
            if i == round:
                continue
            train += data[split * i : split * (i + 1)]
        with open('temp/cross_train.conll', 'w') as f:
            for sent in train:
                if len(sent) == 0:
                    continue
                for wid, (word, ner) in enumerate(sent):
                    f.write('{} {} -- -- {}\n'.format(wid+1, word, ner))
                f.write('\n')
        with open('temp/cross_dev.conll', 'w') as f:
            for sent in dev:
                if len(sent) == 0:
                    continue
                for wid, (word, ner) in enumerate(sent):
                    f.write('{} {} -- -- {}\n'.format(wid+1, word, ner))
                f.write('\n')
        score.append(retrain('temp/cross_train.conll', 'temp/cross_dev.conll'))
        print(score)

if __name__ == '__main__':
    # preprocess()
    
    sample()

    # os.system('cat temp/query.conll >> temp/annotated.conll')
    
    # retrain('temp/annotated.conll', 'temp/annotated.conll')
    # cross_validate()