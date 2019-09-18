import os
import pickle
from stanfordcorenlp import StanfordCoreNLP
from document import *
from ner import *
from nominal import *
from tree import Tree
from collections import deque

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer
from neuronlp2.models import BiRecurrentConvCRF, Embedding, ChainCRF
from neuronlp2 import utils
from sklearn import svm


class OntologyType(object):
    def __init__(self):
        self.word2type = self.load_ontology_word()
        self.uplink = self.load_ontology_hierarchy()
        self.root, self.types = self.contruct_hierarchy()

    def load_ontology_word(self):
        word2type = {}
        with open('ontology/ontology_word.txt', 'r') as f:
            for line in f:
                if len(line) <= 1:
                    continue
                word, type = line.strip().split('\t')
                word2type[word] = type
        return word2type

    def load_ontology_hierarchy(self):
        uplink = {}
        with open('ontology/ontology_hier.txt', 'r') as f:
            for line in f:
                if len(line) <= 1:
                    continue
                type, upper = line.strip().split('\t')
                uplink[type] = upper
        return uplink

    def contruct_hierarchy(self):
        all_types = {}
        for type, upper in self.uplink.items():
            if type not in all_types:
                all_types[type] = Tree(tag=type)
            if upper not in all_types:
                all_types[upper] = Tree(tag=upper)
            all_types[type].parent = all_types[upper]
            all_types[upper].add_child(all_types[type])
        for type in all_types.values():
            if type.parent is None:
                return type, all_types

    def load_decision_tree(self):
        with open('temp/type_decision_tree.dump', 'rb') as f:
            return pickle.load(f)

    def lookup(self, word):
        if word in self.word2type:
            return self.word2type[word]
        if word.lower() in self.word2type:
            return self.word2type[word.lower()]
        if word in self.uplink:
            return self.uplink[word]
        return 'NULL'

    def lookup_all(self, word):
        type = self.lookup(word)
        if type == 'NULL':
            return []
        all_type = [type]
        while type in self.uplink:
            type = self.uplink[type]
            all_type.append(type)
        return all_type


def prepare_train_data(nlp):
    ontology = OntologyType()

    train_set = set()
    for fname in os.listdir('../../data/txt/'):
        if fname.endswith('.dump'):
            train_set.add(fname[:-5])
    print(train_set)

    train_data = []

    for root, dirs, files in os.walk('../../data/ltf/'):
        for file in files:
            if file in train_set:
                print(file)
                sents, doc = read_ltf_offset(os.path.join(root, file))
                for sent in sents:
                    nominals = extract_nominals(sent, nlp, [])
                    for mention in nominals:
                        ont_types = ontology.lookup_all(mention['headword'])
                        if ont_types:
                            train_data.append((sent, mention, ont_types, 1.0))
                            print(sent.get_text())

    with open('temp/type_train_data.dump', 'wb') as f:
        pickle.dump(train_data, f)

def regen_train_data(nlp, fpath):
    ontology = OntologyType()
    decisions = ontology.load_decision_tree()
    network = torch.load('temp/ner_tuned.pt')
    word_alphabet, char_alphabet, pos_alphabet, \
        chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("ner_alphabet/", None)

    train_set = set()
    for fname in os.listdir('../../data/txt/'):
        if fname.endswith('.dump'):
            train_set.add(fname[:-5])
    print(train_set)

    train_data = []
    train_feat = []

    for root, dirs, files in os.walk('../../data/ltf/'):
        for file in files:
            if file in train_set:
                print(file)
                sents, doc = read_ltf_offset(os.path.join(root, file))
                for sent in sents:
                    named_ents, ners, feats = extract_ner(sent)
                    for mention, feat in zip(named_ents, feats):
                        prdt_type = infer_type(feat, decisions)
                        coherence = type_coherence(mention['type'], prdt_type, ontology)
                        if coherence > 0:
                            train_data.append((sent, mention, [prdt_type] + ontology.lookup_all(prdt_type), coherence))
                            train_feat.append((feat, [prdt_type] + ontology.lookup_all(prdt_type), coherence))
                            print(sent.get_text())
                    nominals = extract_nominals(sent, nlp, ners)
                    for mention in nominals:
                        ont_types = ontology.lookup_all(mention['headword'])
                        if ont_types:
                            train_data.append((sent, mention, ont_types, 1.0))
                            with open('tmp', 'w') as f:
                                for i, word in enumerate(sent.words):
                                    f.write('{0} {1} -- -- O\n'.format(i+1, word.word))
                            sent_data = conll03_data.read_data_to_variable('tmp', word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=True)
                            os.system('rm tmp')
                            word, char, pos, chunk, labels, masks, lengths = conll03_data.iterate_batch_variable(sent_data, 1).next()
                            feat = network.feature(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
                            feat_vec = feat[0, mention['head_index'], :]
                            train_feat.append((feat_vec.data.numpy(), ont_types, 1.0))
                            print(sent.get_text())

    with open('temp/' + fpath + 'data.dump', 'wb') as f:
        pickle.dump(train_data, f)
    with open('temp/' + fpath + 'feat.dump', 'wb') as f:
        pickle.dump(train_feat, f)

def extract_features(data_name, feat_name):
    with open('temp/' + data_name, 'rb') as f:
        train_data = pickle.load(f)
    print(len(train_data))

    network = torch.load('temp/ner_tuned.pt')
    word_alphabet, char_alphabet, pos_alphabet, \
        chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("ner_alphabet/", None)

    feats = []
    for sent, mention, ont_types, weight in train_data:
        with open('tmp', 'w') as f:
            for i, word in enumerate(sent.words):
                f.write('{0} {1} -- -- O\n'.format(i+1, word.word))
        sent_data = conll03_data.read_data_to_variable('tmp', word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=True)
        os.system('rm tmp')
        word, char, pos, chunk, labels, masks, lengths = conll03_data.iterate_batch_variable(sent_data, 1).next()
        feat = network.feature(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
        print(feat.size())
        feat_vec = feat[0, mention['head_index'], :]
        feats.append((feat_vec.data.numpy(), ont_types, weight))
        print(np.shape(feats[-1][0]))

    with open('temp/' + feat_name, 'wb') as f:
        pickle.dump(feats, f)

def train_classifier(train_feat):
    ontology = OntologyType()

    with open('temp/' + train_feat, 'rb') as f:
        train_feats = pickle.load(f)
    print(len(train_feats))
    
    clfs = deque()
    type_root = ontology.root
    clfs.append(type_root)
    decisions = []

    while len(clfs):
        upper = clfs.popleft()
        subs = upper.children
        if not subs:
            continue
        print(upper.tag + ' --> ' + ' '.join([sub.tag for sub in subs]))
        for sub in subs:
            clfs.append(sub)

        X, Y, weights = [], [], []
        for feat, ont_type, weight in train_feats:
            for i, sub in enumerate(subs):
                if sub.tag in ont_type:
                    X.append(feat)
                    Y.append(i)
                    weights.append(weight)
                    break
        print(Y, len(Y))
        if filter(lambda x: x != Y[0], Y):  # at least two labels
            svm_clf = svm.SVC(C=1.0)
            svm_clf.fit(X, Y, sample_weight=weights)
            decisions.append((upper, subs, svm_clf))
        elif len(Y) == 0: # no data
            pass
        else:
            decisions.append((upper, subs, Y[0]))

    print(decisions)

    count = 0
    for feat, ont_type, _ in train_feats:
        prdt = infer_type(feat, decisions, 'Entity')
        print(ont_type, prdt, ontology.lookup_all(prdt))
        if ont_type and (prdt == ont_type[0] or prdt in ont_type or ont_type[0] in ontology.lookup_all(prdt)):
            count += 1
            print('*')
    print(count, len(train_feats))

    with open('temp/type_decision_tree.dump', 'wb') as f:
        pickle.dump(decisions, f)

def infer_type(feat, decisions, root='Entity'):
    current = root
    while True:
        found = False
        for upper, subs, clf in decisions:
            if upper.tag == current:
                if isinstance(clf, svm.SVC):
                    subtype = clf.predict(feat.reshape(1, -1))[0]
                    current = subs[subtype].tag
                else:
                    current = subs[clf].tag
                found = True
        if not found:
            return current

def type_coherence(type, fine_grained, ontology):
    if type == 'LOC':
        type = 'Loc'
    elif type == 'ORG':
        type = 'Org'
    elif type == 'WEA':
        type = 'Weapon'
    elif type == 'VEH':
        type = 'Vehicle'

    if type == fine_grained:
        return 1
    if type in ontology.lookup_all(fine_grained) or fine_grained in ontology.lookup_all(type):
        return 0.5
    return 0
    

if __name__ == '__main__':
    # with StanfordCoreNLP('/home/xianyang/stanford-corenlp-full-2017-06-09/') as nlp:
    #     prepare_train_data(nlp)

    # extract_features()

    # train_classifier()

    with StanfordCoreNLP('/home/xianyang/stanford-corenlp-full-2017-06-09/') as nlp:
        regen_train_data(nlp, 'train-2')
    train_classifier('train-2feat.dump')