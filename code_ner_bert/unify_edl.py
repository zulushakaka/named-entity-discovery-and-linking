import json
import sys
import os
from document import read_ltf_offset


def unify_edl_output(ner_file, edl_file):
    edl_output = []
    with open(edl_file, 'r') as f:
        for line in f:
            _, _, mention, span, fb_id, ner, type, _, expanded_mention, ner_stanford, wiki_id = line.strip().split('\t')
            char_begin = int(span[ span.find(':') + 1 : span.find('-') ])
            char_end = int(span[ span.find('-') + 1 : ])
            edl_output.append({'mention': mention, 'char_begin': char_begin, 'char_end': char_end, 'fb_id': fb_id, 'wiki_id': wiki_id, 'ner': ner, 'type': type})

    with open(ner_file, 'r') as f:
        doc = json.load(f)
        for s_id, sent in enumerate(doc):
            for mention in sent['namedMentions']:
                for edl in filter(lambda x: x['type'] == 'NAM', edl_output):
                    char_begin = edl['char_begin']
                    char_end = edl['char_end'] + 1
                    if abs(char_begin - mention['char_begin']) <= 1 and abs(char_end - mention['char_end']) <= 1 and edl['mention'] == mention['mention']:
                        mention['freebaseId'] = edl['fb_id']
                        mention['wikipediaId'] = edl['wiki_id']

            for mention in sent['nominalMentions']:
                for edl in filter(lambda x: x['type'] == 'NOM', edl_output):
                    char_begin = edl['char_begin']
                    char_end = edl['char_end'] + 1
                    if abs(char_begin - mention['head_span'][0]) <= 1 and abs(char_end - mention['head_span'][1]) <= 1 and edl['mention'] == mention['headword']:
                        mention['freebaseId'] = edl['fb_id']
                        mention['wikipediaId'] = edl['wiki_id']

    with open(ner_file, 'w') as f:
        json.dump(doc, f, indent=1, sort_keys=True)


def ltf_to_raw(in_file, out_file):
    _, doc = read_ltf_offset(in_file)
    if doc is None:
        return
    with open(out_file, 'w') as f:
        print(doc)
        f.write(doc.encode('UTF-8'))

if __name__ == '__main__':
    if sys.argv[1] == 'txt':
        in_dir = sys.argv[2]
        out_dir = sys.argv[3]
        for fname in os.listdir(in_dir):
            ltf_to_raw(os.path.join(in_dir, fname), os.path.join(out_dir, fname + '.txt'))
    elif sys.argv[1] == 'unify':
        ner_dir = sys.argv[2]
        edl_dir = sys.argv[3]
        for fname in os.listdir(ner_dir):
            ner_file = os.path.join(ner_dir, fname)
            edl_file = os.path.join(edl_dir, fname[:fname.find('.xml')] + '.txt')
            unify_edl_output(ner_file, edl_file)
    else:
        print('Unknown task: ' + argv[1])