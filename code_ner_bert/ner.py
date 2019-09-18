import sys
import os

sys.path.append(".")
sys.path.append("..")
import nltk
from nltk.corpus import stopwords
eng_sp = set(stopwords.words('english'))
import time
import argparse
import uuid

import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.autograd import Variable

import neuronlp2
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer
from neuronlp2.models import BiRecurrentConvCRF, Embedding, ChainCRF
from neuronlp2 import utils

from gazetteer import lookup_gazetteer, look_gazetteer, lookup_per, lookup_city
import importlib
mod = importlib.import_module("pytorch-pretrained-bert.examples.run_ner")
# from pytorch_pretrained_bert.examples.run_multi_ner import NERPredictor
mod_subtype = importlib.import_module("pytorch-pretrained-bert.examples.run_multi_ner")

def data():
    path = '/media/xianyang/OS/CMU/opera/data/annotation/'
    os.system('rm temp/target.*.conll')
    for fname in os.listdir(path):
        if 'HC00000DW' in fname:
            continue
        os.system('cat {} >> temp/target.train.conll'.format(os.path.join(path, fname)))
    os.system('cp temp/target.train.conll temp/target.test.conll')
    os.system('cat temp/annotated.conll >> temp/target.train.conll')
    os.system('cat temp/old_data.conll >> temp/target.train.conll')
    os.system('cat /media/xianyang/OS/CMU/opera/data/annotation/HC00000DW.ltf.xml.txt.conll >> temp/target.dev.conll')

def main():
    embedding = 'glove'
    embedding_path = '/media/xianyang/OS/workspace/ner/glove.6B/glove.6B.100d.txt'
    word_alphabet, char_alphabet, pos_alphabet, \
    chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("/media/xianyang/OS/workspace/ner/NeuroNLP2/data/alphabets/ner_crf/", None)
    char_dim =30
    num_filters = 30
    window = 3
    mode = 'LSTM'
    hidden_size = 256
    num_layers = 1
    num_labels = ner_alphabet.size()
    tag_space = 128
    p = 0.5
    bigram = True
    embedd_dim = 100
    use_gpu = False

    print(len(word_alphabet.get_content()['instances']))
    print(ner_alphabet.get_content())

    # writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(),
                                 char_dim, char_alphabet.size(),
                                 num_filters, window,
                                 mode, hidden_size, num_layers, num_labels,
                                 tag_space=tag_space, embedd_word=None, p_rnn=p, bigram=bigram)
    network.load_state_dict(torch.load('temp/23df51_model45'))
    
    ner_alphabet.add('B-VEH')
    ner_alphabet.add('I-VEH')
    ner_alphabet.add('B-WEA')
    ner_alphabet.add('I-WEA')
    num_new_word = 0

    with open('temp/target.train.conll', 'r') as f:
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

    print(len(word_alphabet.get_content()['instances']))
    print(ner_alphabet.get_content())
    
    init_embed = network.word_embedd.weight.data
    init_embed = np.concatenate((init_embed, np.zeros((num_new_word, embedd_dim))), axis=0)
    network.word_embedd = Embedding(word_alphabet.size(), embedd_dim, torch.from_numpy(init_embed))
    
    old_crf = network.crf
    new_crf = ChainCRF(tag_space, ner_alphabet.size(), bigram=bigram)
    trans_matrix = np.zeros((new_crf.num_labels, old_crf.num_labels))
    for i in range(old_crf.num_labels):
       trans_matrix[i, i] = 1
    new_crf.state_nn.weight.data = torch.FloatTensor(np.dot(trans_matrix, old_crf.state_nn.weight.data))
    network.crf = new_crf

    target_train_data = conll03_data.read_data_to_variable('temp/target.train.conll', word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=False)
    target_dev_data = conll03_data.read_data_to_variable('temp/target.dev.conll', word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=False)
    target_test_data = conll03_data.read_data_to_variable('temp/target.test.conll', word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=False)

    num_epoch = 50
    batch_size = 32
    num_data = sum(target_train_data[1])
    num_batches = num_data / batch_size + 1
    unk_replace = 0.0
    # optim = SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0, nesterov=True)
    optim = Adam(network.parameters(), lr=1e-3)

    for epoch in range(1, num_epoch+1):
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

            if batch % 20 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time)
                print(log_info)
                num_back = len(log_info)

        writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
        os.system('rm temp/output.txt')
        writer.start('temp/output.txt')
        network.eval()
        for batch in conll03_data.iterate_batch_variable(target_dev_data, batch_size):
            word, char, pos, chunk, labels, masks, lengths, _ = batch
            preds, _, _ = network.decode(word, char, target=labels, mask=masks,
                                          leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
            writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                             preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
        writer.close()

        acc, precision, recall, f1 = evaluate('temp/output.txt')
        log_info = 'dev: %f %f %f %f' % (acc, precision, recall, f1)
        print(log_info)

        if epoch % 10 == 0:
            writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
            os.system('rm temp/output.txt')
            writer.start('temp/output.txt')
            network.eval()
            for batch in conll03_data.iterate_batch_variable(target_test_data, batch_size):
                word, char, pos, chunk, labels, masks, lengths, _ = batch
                preds, _, _ = network.decode(word, char, target=labels, mask=masks,
                                            leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
                writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                                preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
            writer.close()

            acc, precision, recall, f1 = evaluate('temp/output.txt')
            log_info = 'test: %f %f %f %f' % (acc, precision, recall, f1)
            print(log_info)
    

    torch.save(network, 'temp/tuned_0905.pt')
    alphabet_directory = '0905_alphabet/'
    word_alphabet.save(alphabet_directory)
    char_alphabet.save(alphabet_directory)
    pos_alphabet.save(alphabet_directory)
    chunk_alphabet.save(alphabet_directory)
    ner_alphabet.save(alphabet_directory)


def evaluate(output_file):
    score_file = "score"
    os.system("./conll03eval.v2 < %s > %s" % (output_file, score_file))
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


##########
'''
if __name__ != '__main__':
    network = torch.load('temp/tuned_0905.pt')
    word_alphabet, char_alphabet, pos_alphabet, \
        chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("0905_alphabet/", None)
'''
def read_sent_to_variable(sent, word_alphabet, char_alphabet, ner_alphabet, volatile=True):
    word_ids = []
    char_ids = []
    ner_ids = []
    max_char_length = 0
    for token in sent:
        chars = []
        for char in token[0]:
            chars.append(char_alphabet.get_index(char))
        max_char_length = max(max_char_length, len(token[0]))
        word = neuronlp2.io.utils.DIGIT_RE.sub(b"0", token[0])
        ner = token[2]
        word_ids.append(word_alphabet.get_index(word))
        ner_ids.append(ner_alphabet.get_index(ner))
        char_ids.append(chars)
    
    length = len(word_ids)
    char_length = min(neuronlp2.io.utils.MAX_CHAR_LENGTH, max_char_length + neuronlp2.io.utils.NUM_CHAR_PAD)

    wid_inputs = np.empty([1, length], dtype=np.int64)
    wid_inputs[0, :] = word_ids
    cid_inputs = np.empty([1, length, char_length], dtype=np.int64)
    for c, cids in enumerate(char_ids):
        if len(cids) > char_length:
            cids = cids[:char_length]
        cid_inputs[0, c, :len(cids)] = cids
        cid_inputs[0, c, len(cids):] = conll03_data.PAD_ID_CHAR
    nid_inputs = np.empty([1, length], dtype=np.int64)
    nid_inputs[0, :] = ner_ids

    words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
    chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
    ners = Variable(torch.from_numpy(nid_inputs), volatile=volatile)

    return words, chars, ners

def read_result(sent, pred):
    return [ner_alphabet.get_instance(pred[0, _]) for _ in range(len(sent))]

## subtype
subtype_predictor = mod_subtype.NERPredictor()
SUBTYPE_HIERARCHY = {}
SUBTYPE_HIERARCHY['FAC'] = set(['ApartmentBuilding', 'GovernmentBuilding', 'House', 'OfficeBuilding', 'School', 'StoreShop', 'VotingFacility', 
                                'Border', 'Checkpoint', 'Airport', 'MilitaryInstallation', 'TrainStation', 'Barricade', 'Bridge', 'Plaza', 'Tower',
                                'Highway', 'Street'] + ['Building', 'GeographicalArea', 'Installation', 'Structure', 'Way'])
SUBTYPE_HIERARCHY['GPE'] = set(['Country', 'OrganizationOfCountries', 'ProvinceState', 'City', 'Village'] + ['UrbanArea'])
SUBTYPE_HIERARCHY['LOC'] = set(['Address', 'Continent', 'AirSpace', 'CrimeScene', 'Field', 'Neighborhood', 'Region'] + ['GeographicalPosition', 'Land', 'Position'])
SUBTYPE_HIERARCHY['ORG'] = set(['Club', 'Team', 'BroadcastingCompany', 'Corporation', 'Manufacturer', 'NewsAgency', 'CriminalOrganization',
                                'Agency', 'Council', 'FireDepartment', 'LawEnforcementAgency', 'LegislativeBody', 'MonitoringGroup', 'ProsecutorOffice',
                                'Railway', 'Commission', 'GovernmentArmedForce', 'Intelligence', 'NonGovernmentMilitia'] + ['Associaton', 'CommercialOrganization',
                                'CriminalOrganization', 'Government', 'International', 'MilitaryOrganization', 'PoliticalOrganization'])
SUBTYPE_HIERARCHY['PER'] = set(['Mercenary', 'Sniper', 'SportsFan', 'MilitaryOfficer', 'ChiefOfPolice', 'Governor', 'HeadOfGovernment', 'Mayor',
                                'Ambassador', 'Firefighter', 'Journalist', 'Minister', 'Paramedic', 'Scientist', 'Spokesperson', 'Spy', 'ProtestLeader'] + [
                                'Combatant', 'Fan', 'MilitaryPersonnel', 'Politician', 'ProfessionalPosition', 'Protester'])
SUBTYPE_HIERARCHY['VEH'] = set(['Airplane', 'CargoAircraft', 'Helicopter', 'FighterAircraft', 'MilitaryBoat', 'MilitaryTransportAircraft', 'Tank',
                                'Rocket', 'Boat', 'yacht', 'Bus', 'Car', 'FireApparatus', 'Train', 'Truck'] + ['Aircraft' + 'MilitaryVehicle', 'Rocket', 
                                'Watercraft', 'WheeledVehicle'])
SUBTYPE_HIERARCHY['WEA'] = set(['Bomb', 'Grenade', 'Cannon', 'DaggerKnifeSword', 'PoisonGas', 'Artillery', 'Firearm', 'AirToAirMissile', 'AntiAircraftMissile',
                                'Missile', 'SurfaceToAirMissile', 'Rock'] + ['Bomb', 'Bullets', 'Cannon', 'Club', 'DaggerKnifeSword', 'Gas', 
                                'GrenadeLauncher', 'Gun', 'MissleSystem', 'ThrownProjectile'])
#SUBTYPE_HIERARCHY['TTL'] = set([])

def extract_ner(sent):
    try:
        #text = [(word.word, None, 'O') for word in sent.words]
        #words, chars, labels = read_sent_to_variable(text, word_alphabet, char_alphabet, ner_alphabet, volatile=True)
    
        #preds, _, conf = network.decode(words, chars, target=labels, mask=None, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
        #feat = network.feature(words, chars, target=labels, mask=None, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
        #ners = read_result(text, preds)
        #print(sent.get_text())
        ners, ner_probs = mod.pred_ner(sent)
        #print(ners)
        #exit()
        #print(ners)
        subtypes = subtype_predictor.pred_ner(sent)
        #print(subtypes)
    except:
        raise
        return [], [], []

    # with open('tmp', 'w') as f:
    #     for i, word in enumerate(text):
    #         f.write('{0} {1} -- -- O\n'.format(i+1, word.encode('UTF-8')))
    
    # try:
    #     writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    #     writer.start('output.txt')
    #     sent_data = conll03_data.read_data_to_variable('tmp', word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=False, volatile=True)
    #     word, char, pos, chunk, labels, masks, lengths, _ = conll03_data.iterate_batch_variable(sent_data, 1).next()
    #     os.system('rm tmp')
    #     preds, _, _ = network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
    #     feat = network.feature(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
    #     writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
    #                     preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())                                  
    #     writer.close()
    # except:
    #     return [], [], []
    
    # ners = []
    feats = []
    # with open('output.txt', 'r') as ner_file:
    #     for line in ner_file:
    #         if len(line) <= 1:
    #             break
    #         id, word, _, _, _, ner = line.strip().split()
    #         ners.append((int(id)-1, word, ner))
    named_ents = []
    for wid, word in enumerate(sent.words):
        # if word.word == 'Putin':
        #     print(sent.get_text())
        #     print(ners[wid])
        if wid >= len(ners):
            break
        if ners[wid][0] == 'B':
            score = ner_probs[wid]
            if score < 0.6:
                score = 0.6
            type = ners[wid][2:]
            j = wid + 1
            while j < len(sent.words) and j < len(ners) and ners[j][0] == 'I':
                j += 1
            ner_span = (wid, j)
            char_begin = sent.words[wid].begin - 1
            char_end = sent.words[j-1].end
            head_span = [sent.words[j-1].begin-1, sent.words[j-1].end]
            #feats.append(feat[0, j-1, :].data.numpy())
            # if  word.word == 'Putin':
            #     print(type)

            named_ent = {'mention': sent.sub_string(wid, j), 'category': 'NAM', 'type': type, 'subtype': 'n/a', 'subsubtype': 'n/a',
            'char_begin': char_begin, 'char_end': char_end,
            'head_span': head_span, 'headword': sent.words[j-1].word, 'token_span': ner_span, 'score': str(score)}
            if named_ent['mention'] in eng_sp:
                continue
            # if  word.word == 'Putin':
            #     print(named_ent)
            ### gazateer
            gazz = look_gazetteer(named_ent['mention'], named_ent['type'])
            if gazz:
                named_ent['type'] = gazz
            if named_ent['type'] == 'PER':
                per_gazz = lookup_per(named_ent['mention'], named_ent['type'])
                if per_gazz:
                    named_ent['type'] = per_gazz
                    #print(per_gazz)
            if named_ent['type'] == 'GPE':
                if 'russian' in named_ent['mention'].lower() or 'ukrainian' in named_ent['mention'].lower():
                    named_ent['type'] = 'ldcOnt:PER'
                else:
                    city_gazz = lookup_city(named_ent['mention'], named_ent['type'])
                    if city_gazz:
                        named_ent['type'] = city_gazz
            named_ents.append(named_ent)
    # process subtypes
    for span, nertype in subtypes.items():
        if len(nertype) > 10:
            continue # to many types, not trusted
        nertype.sort(key=lambda x: x[1], reverse=True)
        # find if mantches a ner
        match = False
        #overlap = False
        for ner in named_ents:
            if ner['type'].startswith('ldc'):
                continue
            if ner['token_span'][1] == span[1]:
                for subtype, _ in nertype:
                    if ner['type'] != 'TTL' and subtype in SUBTYPE_HIERARCHY[ner['type']]:
                        ner['subtype'] = subtype
                        match = True
                        break
    return named_ents, ners, feats


#if __name__ == '__main__':
    #data()
    #main()
