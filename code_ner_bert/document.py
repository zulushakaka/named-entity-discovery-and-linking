import os
import sys
import xml.etree.ElementTree as ET
import pickle
import stanfordcorenlp
import json


class Word(object):
    def __init__(self, word, begin, end, sent, index):
        self.word = word
        self.begin = begin
        self.end = end
        self.sent = sent
        self.index = index


class Sentence(object):
    def __init__(self, begin, end, index):
        self.begin = begin
        self.end = end
        self.index = index
        self.words = []
        self.annotation = None

    def get_text(self):
        return ' '.join([word.word for word in self.words])

    def sub_string(self, begin, end):
        # return ' '.join([word.word for word in self.words[begin:end]])
        result = ''
        offset = self.words[begin].begin
        for word in self.words[begin:end]:
            for i in range(word.begin - offset - 1):
                result += ' '
            result += word.word
            offset = word.end
        return result

    def get_original_string(self):
        result = ''
        offset = self.words[0].begin
        for word in self.words:
            for i in range(word.begin - offset - 1):
                result += ' '
            result += word.word
            offset = word.end
        return result

    @staticmethod
    def get_original_doc(sents):
        doc = ''.join(['.'] * sents[0].begin)
        #doc = ''
        offset = sents[0].begin - 1
        for sent in sents:
            #print(sent.begin, sent.)
            if sent.begin <= offset:
                doc = doc[:sent.begin - offset - 1]
            for i in range(sent.begin - offset - 1):
                doc += '\n'
            sent_str = sent.get_original_string()
            if '%20' in sent_str:
                sent_str = sent_str.replace('%20', '___')
            doc += sent_str
            offset = sent.end
            #if str.isalnum(sent_str[-1].encode('utf-8')):
            if str.isalnum(sent_str[-1]):
                doc += ';'
                offset += 1
        return doc

    def retokenize(self, nlp):
        try:
            # print(self.get_original_string())
            # print([word.word for word in self.words])
            #nlp_token = nlp.word_tokenize(self.get_text().encode('UTF-8'))
            nlp_token = nlp.word_tokenize(self.get_text())
            # print(nlp_token)
            i, j = 0, 0
            new_words = []
            while i < len(self.words) and j < len(nlp_token):
                word_i = self.words[i].word
                word_j = nlp_token[j]
                if word_j == '-LRB-':
                    word_j = '('
                elif word_j == '-RRB-':
                    word_j = ')'
                elif word_j == '-LSB-':
                    word_j = '['
                elif word_j == '-RSB-':
                    word_j = ']'
                
                if word_i == word_j:
                    new_words.append(self.words[i])
                    i += 1
                    j += 1
                    continue
                elif word_i.startswith(word_j):
                    new_words.append(Word(word_j, self.words[i].begin, self.words[i].begin + len(word_j) - 1, self, 0))
                    self.words[i] = Word(word_i[len(word_j):], self.words[i].begin + len(word_j), self.words[i].end, self, 0)
                    j += 1
                    continue
                elif word_j.startswith(word_i):
                    k = i + 1
                    expanded_word = word_i + self.words[k].word
                    while not expanded_word.startswith(word_j):
                        k += 1
                        expanded_word += self.words[k].word
                    if expanded_word == word_j:
                        new_words.append(Word(expanded_word, self.words[i].begin, self.words[k].end, self, 0))
                        i = k + 1
                        j += 1
                        continue
                    else:
                        tail = len(expanded_word) - len(word_j)
                        new_words.append(Word(expanded_word, self.words[i].begin, self.words[k].end - tail, self, 0))
                        self.words[k] = Word(expanded_word[len(word_j):], self.words[k].end - tail + 1, self.words[k].end, self, 0)
                        i = k
                        j += 1
                        continue
                else:
                    # skip and try to recover
                    new_words.append(self.words[i])
                    i += 1
                    j += 1
                    if i >= len(self.words):
                        break
                    word_i = self.words[i].word
                    word_j = nlp_token[j]
                    if word_i.startswith(word_j) or word_j.startswith(word_i):
                        continue
                    j += 1
                    word_j = nlp_token[j]
                    if word_i.startswith(word_j) or word_j.startswith(word_i):
                        continue
                    print('Error: fail to recover')
                    # raise Exception
                    return

            for wid, word in enumerate(new_words):
                word.index = wid
            self.words = new_words
        except:
            print('Error: fail to recover')
            # raise Exception
            return

def read_raw_text(fname, nlp):
    with open(fname, 'r') as f:
        doc = f.readlines()
        doc = '\n'.join(doc)

    props = {'annotators': 'tokenize, ssplit, ner, parse','pipelineLanguage':'en','outputFormat':'json'}
    print('running raw corenlp for {} ...'.format(fname))
    try:

        nlp_annotation = nlp.annotate(doc.encode('UTF-8'), props)
        nlp_annotation = json.loads(nlp_annotation)
        print('finished corenlp for {}'.format(fname))
        sents = []
        for sent_annot in nlp_annotation['sentences']:
            #print('xiang')
            sent = Sentence(sent_annot['tokens'][0]['characterOffsetBegin']+1, sent_annot['tokens'][-1]['characterOffsetEnd'], sent_annot['index'])
            sent.annotation = {}
            sent.annotation['parse'] = sent_annot['parse']
            ner = []
            for token in sent_annot['tokens']:
                sent.words.append(Word(token['originalText'], token['characterOffsetBegin']+1, token['characterOffsetEnd'], sent, token['index']-1))
                ner.append((token['originalText'], token['ner']))
            sent.annotation['ner'] = ner
            sents.append(sent)
    except:
        print('corenlp error on {}'.format(fname))
        return None, None
    
    return sents, doc

def read_ltf_offset(fname, out_fname=None, nlp=None):
    # sys.stdout.write('parsing {}\n'.format(fname))
    # sys.stdout.flush()
    tree = ET.parse(fname)
    # sys.stdout.write('parsing finished.\n')
    # sys.stdout.flush()
    root = tree.getroot()
    flag = False
    sents = []
    lang = root.attrib['lang']
    if lang != 'eng':
        # sys.stdout.write('Not English!\n')
        # sys.stdout.flush()
        return None, None
    for sent_id, seg in enumerate(root[0][0]):
        text = seg.find('ORIGINAL_TEXT').text
        sent = Sentence(int(seg.attrib['start_char']), int(seg.attrib['end_char']), sent_id)
        tokens = seg.findall('TOKEN')
        for tok_id, token in enumerate(tokens):
            sent.words.append(Word(token.text, int(token.attrib['start_char']), int(token.attrib['end_char']), sent, tok_id))
        # if nlp:
        #     sent.retokenize(nlp)
        # if len(sent.words) < 50:
        #     sents.append(sent)
        sents.append(sent)
        if sent.words[-1].end > 10000 or len(sents) >= 200:
            break
    doc = Sentence.get_original_doc(sents)
    # print(doc.encode('UTF-8'))
    if nlp:
        props = {'annotators': 'tokenize, ssplit, ner, parse','pipelineLanguage':'en','outputFormat':'json'}
        print('running corenlp for {} ...'.format(fname))
        try:
            #print(doc)
            #nlp_annotation = nlp.annotate(doc.encode('UTF-8'), props)
            nlp_annotation = nlp.annotate(doc, props)
            nlp_annotation = json.loads(nlp_annotation, strict=False)
            print('finished corenlp for {}'.format(fname))
            # print(doc)
            # print(nlp_annotation)['sentences'][1]
            # bp = raw_input('bp')
            new_sents = []
            for sent_annot in nlp_annotation['sentences']:
                #print(sent_annot)
                #exit()
                sent = Sentence(sent_annot['tokens'][0]['characterOffsetBegin']+1, sent_annot['tokens'][-1]['characterOffsetEnd'], sent_annot['index'])
                sent.annotation = {}
                sent.annotation['parse'] = sent_annot['parse']
                ner = []
                for token in sent_annot['tokens']:
                    sent.words.append(Word(token['originalText'], token['characterOffsetBegin']+1, token['characterOffsetEnd'], sent, token['index']-1))
                    ner.append((token['originalText'], token['ner']))
                sent.annotation['ner'] = ner
                new_sents.append(sent)
            sents = new_sents
        except Exception as e:
            print(str(e))
            #print(nlp_annotation)
            print('corenlp error on {}; rerun per sentence. ({})'.format(fname, len(sents)))
            #return sents, doc
            for sent in sents:
                # sent.retokenize(nlp)
                # sent.annotation = {}
                # try:
                #     ner = nlp.ner(sent.get_text().encode('UTF-8'))
                #     sent.annotation['ner'] = ner
                # except:
                #     sent.annotation['ner'] = None
                # try:
                #     parse = nlp.parse(sent.get_text().encode('UTF-8'))
                #     sent.annotation['parse'] = parse
                # except:
                #     sent.annotation['parse'] = None
                try:
                    sent_text = sent.get_original_string()
                    props = {'annotators': 'tokenize, ner, parse','pipelineLanguage':'en','outputFormat':'json'}
                    sent_annotation = nlp.annotate(sent_text, props)
                    sent_annotation = json.loads(sent_annotation, strict=False)['sentences'][0]
                    # print(sent_annotation)
                    sent.annotation = {}
                    sent.annotation['parse'] = sent_annotation['parse']
                    ner = []
                    sent_offset = sent.words[0].begin
                    sent.words = []
                    for token in sent_annotation['tokens']:
                        sent.words.append(Word(token['originalText'], sent_offset+token['characterOffsetBegin'], sent_offset+token['characterOffsetEnd']-1, sent, token['index']-1))
                        ner.append((token['originalText'], token['ner']))
                    sent.annotation['ner'] = ner
                except:
                    print('Error: fail to recover')
                    sent.annotation = {}
                    sent.annotation['parse'] = None
                    sent.annotation['ner'] = None

            print('{} done'.format(fname))

    if out_fname:
        with open(out_fname + '.dump', 'wb') as f:
            pickle.dump(sents, f)
        with open(out_fname + '.txt', 'w') as f:
            f.write(doc)
    
    return sents, doc
