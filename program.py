from docria.storage import DocumentIO
from docria.algorithm import dfs
import os
import json
import requests


baseurl = 'https://www.wikidata.org/entity/'
it = 0
persons = []
for filename in os.listdir('svwiki'):
    print(filename)
    if it > 0 and it < 5:
        with DocumentIO.read("svwiki/" + filename) as docria_reader:
            for doc in docria_reader:
                if 'wkd' in doc.props and 'paragraph' in doc.layers and len(doc.layers['paragraph']) > 0 and 'text' in doc.layers['paragraph'][0]:
                    id = 'Q' + str(doc.props['wkd'])
                    title = doc.props['title']
                    firstP = str(doc.layers['paragraph'][0]['text'])
                    person = { 'id': id, 'title': title, 'first_paragraph': firstP}
                    url = baseurl + str(id)
                    r = requests.get(url = url)
                    data = r.json()
                    if 'entities' in data and id in data['entities'] and 'claims' in data['entities'][id] and 'P31' in data['entities'][id]['claims']:
                        for inst in data['entities'][id]['claims']['P31']:
                            if 'mainsnak' in inst and 'datavalue' in inst['mainsnak'] and 'value' in inst['mainsnak']['datavalue'] and 'id' in inst['mainsnak']['datavalue']['value'] and inst['mainsnak']['datavalue']['value']['id'] == 'Q5':
                                persons.append(person)
                                print(title)
                pass
    it = it + 1

with open('dump1.json', 'w') as fp:
    json.dump(persons, fp)
