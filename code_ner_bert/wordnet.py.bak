from nltk.corpus import wordnet as wn
from collections import deque
from threading import Semaphore
import random


# LOCK = Semaphore(1)

def is_hypernym(sem_class, term):
    hypernyms = deque()
    hypernyms.append(term)
    while len(hypernyms) > 0:
        synset = hypernyms.popleft()
        if synset in sem_class:
            return True
        for hyper in synset.hypernyms():
            hypernyms.append(hyper)
    return False


PERSON = set(wn.synsets('person'))
ANIMAL = set(wn.synsets('animal'))
ORGANIZATION = set(wn.synsets('organization') + wn.synsets('military') + wn.synsets('group'))
LOCATION = set(wn.synsets('location'))
GPE = set(wn.synsets('administrative_district'))
FACILITY = set(wn.synsets('facility') + wn.synsets('structure'))
TIME = set(wn.synsets('time') + wn.synsets('date') + wn.synsets('time_period'))
NUMBER = set(wn.synsets('number'))
QUANTITY = set(wn.synsets('definite_quantity'))
MONEY = set(wn.synsets('money'))
PERCENT = set(wn.synsets('percent'))
VEHICLE = set(wn.synsets('vehicle'))
WEAPON = set(wn.synsets('weapon') + wn.synsets('weaponry') + wn.synsets('arms') + wn.synsets('implements_of_war') + wn.synsets('weapons_system') + wn.synsets('munition'))


def get_semantic_class(lemma):
    # rid = random.randint(0, 100)
    # print('wait!%d' % rid)
    # LOCK.acquire()
    # print('get!%d' % rid)

    result = 'UNKNOWN'
    try:
        term = wn.synsets(lemma)
    except:
        print('**' + lemma)
        term = None

    if not term:
        # LOCK.release()
        # print('release!%d' % rid)
        return result

    term = term[0]
    
    if is_hypernym(PERSON, term):
        result = 'PER'
    elif is_hypernym(ORGANIZATION, term):
        result = 'ORG'
    elif is_hypernym(GPE, term):
        result = 'GPE'
    elif is_hypernym(FACILITY, term):
        result = 'FAC'
    elif is_hypernym(LOCATION, term):
        result = 'LOC'
    elif is_hypernym(WEAPON, term):
        result = 'WEA'
    elif is_hypernym(VEHICLE, term):
        result = 'VEH'

    # LOCK.release()
    # print('release!%d' % rid)
    
    return result
