from docria.storage import DocumentIO
from docria.algorithm import dfs
import os
import json
import requests
import bz2

"""
with bz2.open('latest-all.json.bz2', "rt") as bzinput:
    inObj = False #True if we are currently rendering an object
    subObj = 0 #Nr of sub ojbects inside current object
    currObj = '' #All the characters from the current object
    person_data = [] #A list where all objects gets appended to
    job_data = []
    person_dump_iteration = 1
    job_dump_iteration = 1
    for i, line in enumerate(bzinput):
        for c in line:
            if c == '{' and not inObj:
                inObj = True
                currObj = currObj + c
            elif c == '{' and inObj:
                currObj = currObj + c
                subObj = subObj + 1
            elif c == '}' and subObj > 0:
                currObj = currObj + c
                subObj = subObj - 1
            elif c == '}' and subObj == 0:
                inObj = False
                currObj = currObj + c
                try:
                    obj = json.loads(currObj)
                    if 'claims' in obj and 'P31' in obj['claims'] and 'sv' in obj['labels']:
                        for cl in obj['claims']['P31']:
                            if cl['mainsnak']['datavalue']['value']['id'] == 'Q5':
                                person = {}
                                name = obj['labels']['sv']['value']
                                persons_jobs = []
                                if 'P106' in obj['claims']:
                                    for cla in obj['claims']['P106']:
                                        persons_jobs.append(cla['mainsnak']['datavalue']['value']['id'])
                                person['name'] = name
                                person['jobs'] = persons_jobs
                                person_data.append(person)
                            if cl['mainsnak']['datavalue']['value']['id'] == 'Q28640':
                                if 'sv' in obj['labels']:
                                    job = obj['labels']['sv']['value']
                                    job_id = obj['id']
                                    job_data.append({'job': job, 'id': job_id})
                except ValueError as e:
                    print('error parsing json')
                currObj = ''
            elif inObj:
                currObj = currObj + c
        if len(person_data) > 1000:
            filename = 'person_data/person_dump' + str(person_dump_iteration) + '.json'
            person_dump_iteration = person_dump_iteration + 1
            with open(filename, 'w') as fp:
                print('DUMP PERSONS!')
                json.dump(person_data, fp)
                person_data = []
        if len(job_data) > 1000:
            filename = 'job_data/job_dump' + str(job_dump_iteration) + '.json'
            job_dump_iteration = job_dump_iteration + 1
            with open(filename, 'w') as fp:
                print('DUMP JOBS!')
                json.dump(job_data, fp)
                job_data = []
    print(len(person_data))
    print(len(job_data))
    #print(person_data[0]['sitelinks'])
    print(person_data[0].keys())


"""
jobs = {''}
person_dict = {}
for filename in os.listdir('person_data'):
    with open('person_data/' + filename) as f:
        data = json.load(f)
        for person in data:
            person_dict[person['name']] = person
            for job in person['jobs']:
                jobs.add(job)

#print(list(jobs))
jobs = list(jobs)
"""


P21 = Vilket kön, Q6581072 = Kvinna
P569 = Födelsedatum
P19 = Födelseort
P106 = Sysselsättning



files_read = []

with open('./person_data2/read_files') as read_files:
    print(read_files)
    files = read_files.readlines()
    for line in files:
        files_read.append(line[:-1])
print(files_read)

with open('./person_dump.json') as f:
    person_dict = json.load(f)

baseurl = 'https://www.wikidata.org/entity/ '
it = 0
persons = []
person_dump_iteration = 12
for filename in os.listdir('./corpus/enwiki/'):
    print(filename)
    if filename not in files_read:
        print(filename)
        with DocumentIO.read("./corpus/enwiki/" + filename) as docria_reader:
            for doc in docria_reader:
                if 'wkd' in doc.props and 'paragraph' in doc.layers and len(doc.layers['paragraph']) > 0 and 'text' in doc.layers['paragraph'][0]:
                    id = 'Q' + str(doc.props['wkd'])
                    title = doc.props['title']
                    firstP = str(doc.layers['paragraph'][0]['text'])
                    if id in person_dict:
                        person = {'jobs': person_dict[id], 'name': title, 'first_paragraph': firstP, 'id': id}
                        persons.append(person)
                        if len(persons) > 50000:
                            filename = 'person_data2/person_dump' + str(person_dump_iteration) + '.json'
                            person_dump_iteration = person_dump_iteration + 1
                            with open(filename, 'w') as fp:
                                print('DUMP!')
                                json.dump(persons, fp)
                                persons = []
                pass
    #it = it + 1
"""
