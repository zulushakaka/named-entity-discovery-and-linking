import json
import os
import sys
import nltk


def run_ultra_fine_typing(dir):
    prepare_data(dir)
    os.system('mv data_to_run.json open_type-master/release/')
    os.system('cd open_type-master/; python3 main.py RUN_ULTRA_TYPING -lstm_type single -enhanced_mention \
              -data_setup joint -add_crowd -multitask -mode test -reload_model_name release_model -eval_data data_to_run.json -load')
    write_result(dir)
    

def prepare_data(dir):
    json_data = []
    for fname in os.listdir(dir):
        with open(os.path.join(dir, fname), 'r') as f:
            doc = json.load(f)
            for sent in doc:
                doc_id = sent['docID']
                sent_offset = sent['offset']
                raw_sentence = sent['inputSentence']
                for mention in sent['namedMentions'] + sent['nominalMentions']:
                    mention_span = mention['mention']
                    left_context = raw_sentence[:mention['char_begin'] - sent_offset + 1]
                    left_context_token = nltk.word_tokenize(left_context)
                    right_context = raw_sentence[mention['char_end'] - sent_offset + 1:]
                    right_context_token = nltk.word_tokenize(right_context)
                    annot_id = doc_id + ':{}-{}'.format(mention['char_begin'], mention['char_end'])
                    json_data.append({'annot_id': annot_id, 'mention_span': mention_span, 'right_context_token': right_context_token, 'left_context_token': left_context_token, 'y_str': 'none'})
    
    with open('data_to_run.json', 'w') as f:
        for mention in json_data:
            json.dump(mention, f)
            f.write('\n')

def write_result(dir):
    types = {}
    with open('open_type-master/release_model.json') as f:
        pred_dict = json.load(f)
        for id, pred in pred_dict.items():
            types[id] = pred['pred']

    for fname in os.listdir(dir):
        with open(os.path.join(dir, fname), 'r') as f:
            doc = json.load(f)
        for sent in doc:
            doc_id = sent['docID']
            for mention in sent['namedMentions'] + sent['nominalMentions']:
                annot_id = doc_id + ':{}-{}'.format(mention['char_begin'], mention['char_end'])
                fine_types = types[annot_id]
                mention['fineGrainedType'] = fine_types
        with open(os.path.join(dir, fname), 'w') as f:
            json.dump(doc, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    dir = os.sys.argv[1]
    print('processing folder {}'.format(dir))
    run_ultra_fine_typing(dir)