from dictionary import is_url


title_list = set()
with open('gazetteer/jobtitles.lst', 'r') as f:
    for line in f:
        title_list.add(line.strip().lower())
title_list.add('president')

def extract_filler(sent, nlp, ners):
    titles = extract_title(sent, nlp, ners)
    times = extract_time(sent, nlp)
    numericals = extract_numerical(sent, nlp)
    urls = extract_url(sent)
    all_fillers = titles + times + numericals + urls

    return all_fillers

def extract_title(sent, nlp, ners):
    titles = []
    for wid, word in enumerate(sent.words):
        found = False
        if word.word.lower() in title_list:
            title = {'mention': word.word, 'char_begin': word.begin-1, 'char_end': word.end, 'head_span': [word.begin-1, word.end], 'type': 'TITLE'}
            found = True
        elif wid + 1 < len(sent.words):
            text = sent.sub_string(wid, wid+2)
            if text.lower() in title_list:
                title = {'mention': text, 'char_begin': sent.words[wid].begin-1, 'char_end': sent.words[wid+1].end, 'head_span': [sent.words[wid+1].begin-1, sent.words[wid+1].end], 'type': 'TITLE'}
                found = True
            elif wid + 2 < len(sent.words):
                text = sent.sub_string(wid, wid+3)
                if text.lower() in title_list:
                    title = {'mention': text, 'char_begin': sent.words[wid].begin-1, 'char_end': sent.words[wid+2].end, 'head_span': [sent.words[wid+2].begin-1, sent.words[wid+2].end], 'type': 'TITLE'}
                    found = True
        if found:
            valid = False
            for ner in ners:
                if ner == 'B-PER':
                    valid = True
                    break
            if valid:
                titles.append(title)
    return titles

def extract_time(sent, nlp):
    # ners = nlp.ner(sent.get_text().encode('UTF-8'))
    ners = nlp['ner']
    if ners is None:
        return []
    if len(ners) != len(sent.words):
        return []
    time = []
    tmp = []
    for wid, (word, ner) in enumerate(ners):
        if ner == 'DATE' or ner == 'TIME':
            tmp.append(wid)
        elif tmp:
            text = sent.sub_string(tmp[0], tmp[-1]+1)
            begin_offset = sent.words[tmp[0]].begin
            # print(len(sent.words), tmp)
            end_offset = sent.words[tmp[-1]].end
            time.append({'mention': text, 'char_begin': begin_offset-1, 'char_end': end_offset, 'head_span': [sent.words[tmp[-1]].begin-1, end_offset], 'type': 'TIME', 'score':'0.9'})
            tmp = []
    
    return time

def extract_numerical(sent, nlp):
    ners = nlp['ner']
    if ners is None:
        return []
    if len(ners) != len(sent.words):
        return []
    num = []
    tmp = []
    for wid, (word, ner) in enumerate(ners):
        if ner == 'NUMBER' or ner == 'PERCENT':
            tmp.append(wid)
        elif tmp:
            text = sent.sub_string(tmp[0], tmp[-1]+1)
            begin_offset = sent.words[tmp[0]].begin
            # print(len(sent.words), tmp)
            end_offset = sent.words[tmp[-1]].end
            num.append({'mention': text, 'char_begin': begin_offset-1, 'char_end': end_offset, 'head_span': [sent.words[tmp[-1]].begin-1, end_offset], 'type': 'NUMERICAL', 'score':'0.9'})
            tmp = []
    
    return num

def extract_url(sent):
    urls = []
    for word in sent.words:
        if is_url(word.word):
            urls.append({'mention': word.word, 'char_begin': word.begin-1, 'char_end': word.end, 'head_span': [word.begin-1, word.end], 'type': 'URL', 'score': '0.9'})
    return urls
