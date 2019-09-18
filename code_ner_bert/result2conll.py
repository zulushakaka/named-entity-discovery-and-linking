import json


def json2conll(gold_file, json_file):
    with open(json_file, 'r') as f:
        json_doc = json.load(f)
    for sent in json_doc:
        text = sent['inputSentence']
        text = text.strip().split()
        for i, word in enumerate(text):
            print('{} {} -- -- O O'.format(i+1, word))
        print('')

def swap(fname):
    with open(fname, 'r') as f:
        for line in f:
            if len(line.strip()) <= 1:
                continue
            # print(line)
            id, word, _, _, out, gold = line.strip().split(' ')
            print('{} {} -- -- {} {}'.format(id, word, gold, out))
        print('')

def output2conll(json_file):
    with open(json_file, 'r') as f:
        json_doc = json.load(f)
    for sent in json_doc:
        text = sent['inputSentence']
    


if __name__ == '__main__':
    # json2conll(None, 'output/HC00002Y7.ltf.xml.json')
    import sys

    swap(sys.argv[1])