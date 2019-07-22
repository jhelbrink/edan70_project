from docria.storage import DocumentIO
import os
import json

with open('./person_dump.json') as f:
    person_dict = json.load(f)

persons = []
person_dump_iteration = 0
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
                            filename = 'person_data/person_dump' + str(person_dump_iteration) + '.json'
                            person_dump_iteration = person_dump_iteration + 1
                            with open(filename, 'w') as fp:
                                print('DUMP!')
                                json.dump(persons, fp)
                                persons = []
                pass
