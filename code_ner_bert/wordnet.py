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
PERSON_COMBATANT = set(wn.synsets('combatant'))
PERSON_MERCENARY = set(wn.synsets('mercenary'))
PERSON_SNIPER = set(wn.synsets('sniper'))
PERSON_FAN = set(wn.synsets('fan'))
PERSON_POLICE = set(wn.synsets('police'))
PERSON_POLITICIAN = set(wn.synsets('politician'))
PERSON_AMBASSADOR = set(wn.synsets('ambassador'))
PERSON_FIREFIGHTER = set(wn.synsets('firefighter'))
PERSON_JOURNALIST = set(wn.synsets('journalist'))
PERSON_MINISTER = set(wn.synsets('minister'))
PERSON_PARAMEDIC = set(wn.synsets('paramedic'))
PERSON_SCIENTIST = set(wn.synsets('scientist'))
PERSON_SPOKEPERSON = set(wn.synsets('spokeperson'))
PERSON_SPY = set(wn.synsets('spy'))
PERSON_PROTESTER = set(wn.synsets('protester'))

ANIMAL = set(wn.synsets('animal'))

ORGANIZATION = set(wn.synsets('organization') + wn.synsets('military') + wn.synsets('group'))
ORGANIZATION_GOVERNMENT = set(wn.synsets('government'))
ORGANIZATION_POLITICAL = set(wn.synsets('party') + wn.synsets('court'))
ORGANIZATION_MILITARY = wn.synsets('military')

LOCATION = set(wn.synsets('location'))
GPE = set(wn.synsets('administrative_district'))
FACILITY = set(wn.synsets('facility') + wn.synsets('structure'))
TIME = set(wn.synsets('time') + wn.synsets('date') + wn.synsets('time_period'))
NUMBER = set(wn.synsets('number'))
QUANTITY = set(wn.synsets('definite_quantity'))
MONEY = set(wn.synsets('money'))
PERCENT = set(wn.synsets('percent'))

VEHICLE = set(wn.synsets('vehicle'))
VEHICLE_AIRCRAFT = set(wn.synsets('aircraft'))
VEHICLE_ROCKET = set(wn.synsets('rocket'))
VEHICLE_WATERCRAFT = set(wn.synsets('watercraft'))
VEHICLE_BUS = set(wn.synsets('bus'))
VEHICLE_CAR = set(wn.synsets('car'))
VEHICLE_TRAIN = set(wn.synsets('train'))
VEHICLE_TRUCK = set(wn.synsets('truck'))

WEAPON = set(wn.synsets('weapon') + wn.synsets('weaponry') + wn.synsets('arms') + wn.synsets('implements_of_war') + wn.synsets('weapons_system') + wn.synsets('munition'))

CRIME = set(wn.synsets('crime'))

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

def get_semantic_class_with_subtype(lemma):
    type = 'n/a'
    subtype, subsubtype = 'n/a', 'n/a'

    term = wn.synsets(lemma)
    if not term:
        return type, subtype, subsubtype

    term = term[0]
    #print(lemma)
    # if lemma == 'police':
    #     print('police xiangk')

    # if is_hypernym(PERSON, term):
    #     type = 'PER'
    #     if is_hypernym(PERSON_COMBATANT, term):
    #         subtype = 'Combatant'
    #         if is_hypernym(PERSON_MERCENARY, term):
    #             subsubtype = 'Mercenary'
    #         elif is_hypernym(PERSON_SNIPER, term):
    #             subsubtype = 'Sniper'
    #     elif is_hypernym(PERSON_FAN, term):
    #         subtype, subsubtype = 'Fan', 'n/a'
    #     elif is_hypernym(PERSON_POLICE, term):
    #         subtype, subsubtype = 'Police', 'n/a'
    #     elif is_hypernym(PERSON_POLITICIAN, term):
    #         subtype, subsubtype = 'Politician', 'n/a'
    #     elif is_hypernym(PERSON_PROTESTER, term):
    #         subtype, subsubtype = 'Protester', 'n/a'
    #     elif is_hypernym(PERSON_AMBASSADOR, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Ambassador'
    #     elif is_hypernym(PERSON_FIREFIGHTER, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Firefighter'
    #     elif is_hypernym(PERSON_JOURNALIST, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Journalist'
    #     elif is_hypernym(PERSON_MINISTER, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Minister'
    #     elif is_hypernym(PERSON_PARAMEDIC, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Paramedic'
    #     elif is_hypernym(PERSON_SCIENTIST, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Scientist'
    #     elif is_hypernym(PERSON_SPOKEPERSON, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Spokeperson'
    #     elif is_hypernym(PERSON_SPY, term):
    #         subtype, subsubtype = 'ProfessionalPosition', 'Spy'
    if is_hypernym(PERSON, term):
        type = 'PER'
    if is_hypernym(PERSON_COMBATANT, term):
        type = 'PER'
        subtype = 'Combatant'
        if is_hypernym(PERSON_MERCENARY, term):
            subsubtype = 'Mercenary'
        elif is_hypernym(PERSON_SNIPER, term):
            subsubtype = 'Sniper'
    elif is_hypernym(PERSON_FAN, term):
        type = 'PER'
        subtype, subsubtype = 'Fan', 'n/a'
    elif is_hypernym(PERSON_POLICE, term):
        type = 'PER'
        subtype, subsubtype = 'Police', 'n/a'
    elif is_hypernym(PERSON_POLITICIAN, term):
        type = 'PER'
        subtype, subsubtype = 'Politician', 'n/a'
    elif is_hypernym(PERSON_PROTESTER, term):
        type = 'PER'
        subtype, subsubtype = 'Protester', 'n/a'
    elif is_hypernym(PERSON_AMBASSADOR, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Ambassador'
    elif is_hypernym(PERSON_FIREFIGHTER, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Firefighter'
    elif is_hypernym(PERSON_JOURNALIST, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Journalist'
    elif is_hypernym(PERSON_MINISTER, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Minister'
    elif is_hypernym(PERSON_PARAMEDIC, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Paramedic'
    elif is_hypernym(PERSON_SCIENTIST, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Scientist'
    elif is_hypernym(PERSON_SPOKEPERSON, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Spokeperson'
    elif is_hypernym(PERSON_SPY, term):
        type = 'PER'
        subtype, subsubtype = 'ProfessionalPosition', 'Spy'
    if subtype != 'n/a' or subsubtype != 'n/a':
        return type, subtype, subsubtype
    if is_hypernym(ORGANIZATION, term):
        type = 'ORG'
    if is_hypernym(ORGANIZATION_GOVERNMENT, term):
        type = 'ORG'
        subtype, subsubtype = 'Government', 'n/a'
    elif is_hypernym(ORGANIZATION_POLITICAL, term):
        type = 'ORG'
        subtype, subsubtype = 'PoliticalOrganization', 'n/a'
    elif is_hypernym(ORGANIZATION_MILITARY, term):
        type = 'ORG'
        subtype, subsubtype = 'MilitaryOrganization', 'n/a'
    if subtype != 'n/a' or subsubtype != 'n/a':
        return type, subtype, subsubtype
    if is_hypernym(GPE, term):
        type = 'GPE'

    if is_hypernym(FACILITY, term):
        type = 'FAC'

    if is_hypernym(LOCATION, term):
        type = 'LOC'

    if is_hypernym(WEAPON, term):
        type = 'WEA'

    if is_hypernym(VEHICLE, term):
        type = 'VEH'
    if is_hypernym(VEHICLE_AIRCRAFT, term):
        type = 'VEH'
        subtype, subsubtype = 'Aircraft', 'n/a'
    elif is_hypernym(VEHICLE_ROCKET, term):
        type = 'VEH'
        subtype, subsubtype = 'Rocket', 'n/a'
    elif is_hypernym(VEHICLE_WATERCRAFT, term):
        type = 'VEH'
        subtype, subsubtype = 'Watercraft', 'n/a'
    elif is_hypernym(VEHICLE_BUS, term):
        type = 'VEH'
        subtype, subsubtype = 'WheeledVehicle', 'Bus'
    elif is_hypernym(VEHICLE_CAR, term):
        type = 'VEH'
        subtype, subsubtype = 'WheeledVehicle', 'Car'
    elif is_hypernym(VEHICLE_TRAIN, term):
        type = 'VEH'
        subtype, subsubtype = 'WheeledVehicle', 'Train'
    elif is_hypernym(VEHICLE_TRUCK, term):
        type = 'VEH'
        subtype, subsubtype = 'WheeledVehicle', 'Truck'
    if subtype != 'n/a' or subsubtype != 'n/a':
        return type, subtype, subsubtype
    if is_hypernym(CRIME, term):
        type = 'CRM'

    return type, subtype, subsubtype
