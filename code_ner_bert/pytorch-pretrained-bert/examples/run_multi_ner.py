import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
# from run_ner import convert_examples_to_features, read_ner_example, read_sent

import os
# import wget
import json
import random
import logging
import argparse
from tqdm import tqdm, trange
import numpy as np
from collections import defaultdict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

## utility
class NERExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask,
        segment_ids, label_id, label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_mask = label_mask

def read_sent(sent):
    examples = []
    example = ''
    label = ''
    idx = 0
    for i, w in enumerate(sent.words):
        example = ' '.join((example, w.word))
        label = ' '.join((label, 'O'))
    guid = str(idx)
    examples.append(NERExample(guid, example.strip(), label.strip()))
    return examples

def read_ner_example(file, limit=None, no_label=False):
    logger.info("LOOKING AT {}".format(file))
    examples = []
    with open(file, encoding='utf-8') as f:
        example = ''
        label = ''
        idx = 0
        for i, line in enumerate(f):
            if len(line) == 1:
                guid = str(idx)
                idx += 1
                examples.append(
                    NERExample(guid, example.strip(), label.strip()))
                example = ''
                label = ''
                continue
            if not(limit is None) and idx == limit:
                break
            line_split = line.split()
            example = ' '.join((example, line_split[1]))
            if no_label:
                label = ' '.join((label, 'O'))
            else:
                label = ' '.join((label, line_split[4]))
            # if line_split[4] not in get_labels():
            #     print(line_split[4])
    return examples

def tokenize_label(text, token_text, label):
    text = text.split()
    label = label.split()
    if len(text) > len(label):
        label += ['O'] * (len(text) - len(label))
    elif len(text) < len(label):
        label = label[:len(text)]
    #assert len(text) == len(label)
    token_label = []
    idx = 0
    token_text_no = []
    for tt in token_text:
        if not tt.startswith('##'):
            token_text_no.append(tt)
    if len(token_text_no) != len(text):
        print(token_text_no)
        print(text)
        print(label)

    for tt in token_text:
        
        if tt.startswith('##'):
            token_label.append('X')
        else:
            token_label.append(label[idx])
            idx += 1
    assert idx == len(label)
    assert len(token_label) == len(token_text)
    return token_label

def convert_labels_to_idx(labels, label_map):
    labels_ids = []
    label_mask = []
    for l in labels:
        if l == 'X':
            labels_ids.append(-1)
            label_mask.append(0)
        else:
            if l not in label_map:
                logger.info("label %s not in label map" %(l))
            labels_ids.append(label_map[l])
            label_mask.append(1)
    return labels_ids, label_mask

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        #print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)
        labels_a = tokenize_label(example.text_a, tokens_a, example.label)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            labels_a = labels_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        labels = ['X'] + labels_a + ['X']
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids, label_mask = convert_labels_to_idx(labels, label_map)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_ids += padding
        label_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        #label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s " % " ".join(
                    [str(x) for x in labels]))
            logger.info("label_ids: %s" % " ".join(
                    [str(x) for x in label_ids]))
            logger.info("label_mask: %s" % " ".join(
                    [str(x) for x in label_mask]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              label_mask=label_mask))
    return features

## model
class BertForTokenMultiClassification(BertPreTrainedModel):
    """
    Multi-task token classification. each task has labels B, I, O
    """
    def __init__(self, config, num_tasks):
        super(BertForTokenMultiClassification, self).__init__(config)
        self.num_tasks = num_tasks
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, 3) for _ in range(num_tasks)])
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, label_masks=None, task=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        
        if task is None:
            logits = [classifier(sequence_output) for classifier in self.classifiers]
            return logits

        logits = self.classifiers[task](sequence_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if label_masks is not None:
                active_loss = label_masks.view(-1) == 1
                active_logits = logits.view(-1, 3)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return loss
        else:
            return logits

## data
def get_data(path, fine_grain_id):
    print('downloading {} data'.format(fine_grain_id))
    url = 'https://blender04.cs.rpi.edu/~panx2/tmp/fine-grained/{0}/{0}.sent.json'.format(fine_grain_id)
    print(url)
    wget.download(url, out='{}/{}.sent.json'.format(path, fine_grain_id))
    with open('{}/{}.sent.json'.format(path, fine_grain_id), 'r') as f:
        with open('{}/{}.sent.conll'.format(path, fine_grain_id), 'w') as out:
            for line in f:
                json_sent = json.loads(line)
                sentences = json_sent['sentences']
                for sentence in sentences:
                    for t_id, token in enumerate(sentence):
                        word = token[0][0]
                        label = token[1]
                        out.write('{} {} -- -- {}\n'.format(t_id+1, word, label))
                    out.write('\n')

class MultiClassDataLoader(DataLoader):
    def __init__(self, cache_path, mapping_path, cluster_path, tokenizer, batch_size=32, shuffle=True, limit=2000):
        # read mapping file
        self.mapping = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                opera_type = tokens[0]
                if opera_type == '':
                    continue
                fine_grained_types = tokens[1:tokens.index('') if '' in tokens else len(tokens)]
                if len(fine_grained_types) == 0:
                    continue
                self.mapping[opera_type] = fine_grained_types
        print(self.mapping)
        self.clusters = []
        with open(cluster_path, 'r') as f:
            for line in f:
                types = line.strip().split()
                self.clusters.append(set(types))
        print(self.clusters)
        self.typeneighbours = {}
        for operatype, finegrained in self.mapping.items():
            for cluster in self.clusters:
                if operatype in cluster:
                    neighbours = []
                    for otheropera in cluster-set(operatype):
                        if otheropera in self.mapping:
                            neighbours.extend(self.mapping[otheropera])
                    for fg in finegrained:
                        self.typeneighbours[fg] = list(filter(lambda x: x != fg, neighbours))
                    break
        print(self.typeneighbours)

        # retrieve training data
        cached_data = os.listdir(cache_path)
        self.features = {}
        self.neg_features = {}
        for finegrained in self.mapping.values():
            for t in finegrained:
                tfile = t + '.sent.conll'
                if tfile not in cached_data:
                    get_data(cache_path, t)
                if t in self.features:
                    continue
                data_example = read_ner_example(os.path.join(cache_path, tfile), limit=limit)
                print(len(data_example))
                feature = convert_examples_to_features(data_example, ['B', 'I', 'O'], 100, tokenizer)
                neg_data_example = read_ner_example(os.path.join(cache_path, tfile), limit=limit, no_label=True)
                neg_feature = convert_examples_to_features(neg_data_example, ['B', 'I', 'O'], 100, tokenizer)
                self.features[t] = feature
                self.neg_features[t] = neg_feature
        print(self.features.keys())

        self.tasks = list(self.features.keys())
        self.num_tasks = len(self.tasks)
        self.batch_size = batch_size
        self.batches = []
        self.total_batches = []
        self.shuffle = shuffle

        for i, task in enumerate(self.tasks):
            task_size = len(self.features[task])
            task_batch_num = (task_size + batch_size - 1) // batch_size
            task_batches = [(i, j * batch_size, (j+1) * batch_size) for j in range(task_batch_num-1)]
            task_batches.append((i, (task_batch_num-1) * batch_size, task_size))
            self.batches.extend(task_batches)

    def negative_sampling(self):
        # add negative examples
        print('negative sampling...')
        self.neg_batches = []
        for i, task in enumerate(self.tasks):
            task_size = len(self.features[task])
            task_batch_num = (task_size + self.batch_size - 1) // self.batch_size
            for _ in range(task_batch_num):
                negative_batch = random.choice(self.batches)
                while negative_batch[0] == i:
                    negative_batch = random.choice(self.batches)
                self.neg_batches.append((i, negative_batch[0], negative_batch[1], negative_batch[2]))
            if task in self.typeneighbours:
                neighbour_tasks = self.typeneighbours[task]
                if len(neighbour_tasks) < 3:
                    continue
                print('sample from neighbours of {}...'.format(task))
                for _ in range(task_batch_num):
                    negative_batch = random.choice(self.batches)
                    while negative_batch[0] == i or self.tasks[negative_batch[0]] not in neighbour_tasks:
                        negative_batch = random.choice(self.batches)
                    self.neg_batches.append((i, negative_batch[0], negative_batch[1], negative_batch[2]))
        print(len(self.batches), len(self.neg_batches))
        self.total_batches = self.batches + self.neg_batches

    def __iter__(self):
        self.negative_sampling()
        if self.shuffle:
            random.shuffle(self.total_batches)
        for batch in self.total_batches:
            if len(batch) == 3:
                task, begin, end = batch
                batch_features = self.features[self.tasks[task]][begin:end]
                input_ids = torch.tensor([f.input_ids for f in batch_features], dtype=torch.long)
                input_mask = torch.tensor([f.input_mask for f in batch_features], dtype=torch.long)
                segment_ids = torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long)
                label_ids = torch.tensor([f.label_id for f in batch_features], dtype=torch.long)
                label_masks = torch.tensor([f.label_mask for f in batch_features], dtype=torch.long)
                yield input_ids, input_mask, segment_ids, label_ids, label_masks, task
            else:
                task, neg_task, begin, end = batch
                batch_features = self.neg_features[self.tasks[neg_task]][begin:end]
                input_ids = torch.tensor([f.input_ids for f in batch_features], dtype=torch.long)
                input_mask = torch.tensor([f.input_mask for f in batch_features], dtype=torch.long)
                segment_ids = torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long)
                label_ids = torch.tensor([f.label_id for f in batch_features], dtype=torch.long)
                label_masks = torch.tensor([f.label_mask for f in batch_features], dtype=torch.long)
                yield input_ids, input_mask, segment_ids, label_ids, label_masks, task

    def __len__(self):
        return max(len(self.total_batches), len(self.batches)*3)

## pipeline
class NERPredictor(object):
    def __init__(self, model_path='./OPERANER/cased_5.pt', 
                    config_path='./OPERANER/config.json',
                    task_path='./OPERANER/tasks.txt', 
                    mapping_path='./OPERANER/mapping.csv',
                    hier_path='./OPERANER/ontology.csv',
                    device='cpu'):
        print('creating NER object!')
        self.task_list = []
        with open(task_path, 'r') as f:
            for line in f:
                _, task = line.strip().split()
                self.task_list.append(task)
        self.mapping = {}
        self.inverse_mapping = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                opera_type = tokens[0]
                if opera_type == '':
                    continue
                fine_grained_types = tokens[1:tokens.index('') if '' in tokens else len(tokens)]
                if len(fine_grained_types) == 0:
                    continue
                self.mapping[opera_type] = fine_grained_types
                for fine_grained_type in fine_grained_types:
                    self.inverse_mapping[fine_grained_type] = opera_type
        self.subtype_hierarchy = {}
        with open(hier_path, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                if tokens[0].startswith('LDC_ent_'):
                    type, subtype, subsubtype = tokens[1], tokens[3], tokens[5]
                    self.subtype_hierarchy[subsubtype] = subtype
        self.device = device
        self.model = BertForTokenMultiClassification(BertConfig(config_path), len(self.task_list))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)

    def pred_ner(self, sent):
        #print(sent.get_text())
        eval_examples = read_sent(sent)
        label_list = ['B', 'I', 'O']
        eval_features = convert_examples_to_features(
                eval_examples, ['B', 'I', 'O'], 350, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_masks = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
            all_segment_ids, all_label_ids, all_label_masks)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        # eval_loss, eval_accuracy = 0, 0
        # nb_eval_steps, nb_eval_examples = 0, 0
        pred_lists = []
        pred_logits = []
        for input_ids, input_mask, segment_ids, label_ids, label_masks in eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            label_masks = label_masks.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
            for task, logit in zip(self.task_list, logits):
                active_loss = label_masks.view(-1) == 1
                active_logit = F.softmax(logit.view(-1, 3)[active_loss], dim=1)
                active_logit = active_logit.detach().cpu().numpy()
                active_preds = np.argmax(active_logit, axis=1)
                # print(active_preds)
                opera_type = self.inverse_mapping[task]
                pred_ner_list = [label_list[p] for p in active_preds]
                # print(pred_ner_list)
                if all([pred != 'B' for pred in pred_ner_list]):
                    continue
                pred_lists.append((pred_ner_list, opera_type))
                pred_logits.append(active_logit)
                # print(pred_ner_list, opera_type)
            break #only one test sample

        # combine and rank
        predicted_ners = defaultdict(list)
        for i, (pred_ner_list, opera_type) in enumerate(pred_lists):
            begin = 0
            while begin < len(pred_ner_list):
                if pred_ner_list[begin] == 'B':
                    end = begin +1
                    while end < len(pred_ner_list) and pred_ner_list[end] == 'I':
                        end += 1
                    predicted_ners[(begin, end)].append((opera_type, pred_logits[i][begin][0]))
                    begin = end
                else:
                    begin += 1

        # voting
        for span, subsubtypes in predicted_ners.items():
            votes = defaultdict(int)
            for subsubtype in subsubtypes:
                if subsubtype[0] in self.subtype_hierarchy:
                    votes[self.subtype_hierarchy[subsubtype[0]]] += 1
                else:
                    votes[subsubtype[0]] += 1
            best_subtype, max_vote = None, 0
            for subtype, vote in votes.items():
                if vote > max_vote:
                    max_vote = vote
                    best_subtype = subtype
            predicted_ners[span] = [(best_subtype, max_vote / sum(votes.values()))]
        # print(predicted_ners)
        return predicted_ners


def train():
    NUM_EPOCH = 5
    learning_rate = 3e-5
    warmup_proportion = 0.1

    # prepare data
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
    dataloader = MultiClassDataLoader('../../data/', '../../mapping.csv', '../../typecluster.txt', tokenizer, batch_size=64)
    num_train_optimization_steps = len(dataloader) * NUM_EPOCH
    print('# of batches:', len(dataloader))

    # prepare model
    model = BertForTokenMultiClassification.from_pretrained('bert-base-cased', cache_dir='../../', num_tasks=dataloader.num_tasks)
    model.cuda()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

    print(dataloader.tasks)
    with open('../../tasks.txt', 'w') as f:
        for i, t in enumerate(dataloader.tasks):
            f.write('{} {}\n'.format(i, t))

    iter_count = 0
    # torch.save(model.state_dict(), '../../epoch_0.pt')
    with open('../../config.json', 'w') as f:
        f.write(model.config.to_json_string())

    for epoch in range(NUM_EPOCH):
        model.train()
        train_loss = 0

        for batch_id, batch in enumerate(tqdm(dataloader, desc='batch')):
            optimizer.zero_grad()

            features, task = batch[:-1], batch[-1]
            features = tuple(t.cuda() for t in features)
            input_ids, input_mask, segment_ids, label_ids, label_masks = features
            loss = model(input_ids, segment_ids, input_mask, label_ids, label_masks, task)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if (batch_id + 1) % 200 == 0:
                print(train_loss / (batch_id + 1))

        print('epoch {} finished.'.format(epoch+1))
        torch.save(model.state_dict(), '../../cased_{}.pt'.format(epoch+1))

class Word:
    def __init__(self, word):
        self.word = word
class Sent:
    def __init__(self, words):
        self.words = words
    def get_text(self):
        return ' '.join([w.word for w in self.words])
def run_eval(eval_conll, limit=None):
    predictor = NERPredictor()
    idx = 0
    with open(eval_conll, 'r') as f:
        example = []
        for i, line in enumerate(f):
            if len(line) == 1:
                idx += 1
                sent = Sent(example)
                predictor.pred_ner(sent)
                example = []
                continue
            if not(limit is None) and idx == limit:
                break
            line_split = line.split()
            example.append(Word(line_split[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--file', type=str)
    parser.add_argument('--limit', type=int, default=100)

    args = parser.parse_args()
    if args.train:
        train()
    elif args.eval:
        run_eval(args.file, args.limit)
