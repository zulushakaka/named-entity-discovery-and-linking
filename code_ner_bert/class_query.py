import os
import sys
import json


def query(dir, type):
    files = os.listdir(dir)
    for fname in files:
        if fname.endswith('.json'):
            with open(os.path.join(dir, fname), 'r') as f:
                doc = json.load(f)
                for sent in doc:
                    for mention in sent['namedMentions'] + sent['nominalMentions']:
                        if mention['type'] == type:
                            print(mention['@id'], mention['mention'])

def filler(dir):
    files = os.listdir(dir)
    for fname in files:
        if fname.endswith('.json'):
            with open(os.path.join(dir, fname), 'r') as f:
                doc = json.load(f)
                for sent in doc:
                    for mention in sent['fillerMentions']:
                        print(mention)


if __name__ == '__main__':
    dir = sys.argv[1]
    type = sys.argv[2]
    if type == 'filler':
        filler(dir)
    else:
        query(dir, type)