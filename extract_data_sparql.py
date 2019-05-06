
import requests
import pandas as pd
import json

pd.options.display.max_rows = 1000000
pd.options.display.max_columns = 80
pd.options.display.width = 200

prefixes = '''PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
'''

english_query = '''
SELECT *  
WHERE
{
	?human wdt:P31 wd:Q5
	; wdt:P106 ?occupation .
}

LIMIT 1000000'''


url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
headers = {'Accept': 'application/json'}
#had to use the flag 'strict: false' in order to escape the \n and \r
data = requests.get(url, params={'query': prefixes + english_query, 'format': 'json'}).json()
profession = []
for item in data['results']['bindings']:
    splittedPerson = item.get('human', {}).get('value').split('/')
    splittedProfession = item.get('occupation', {}).get('value').split('/')

    profession.append({
        'id': splittedPerson[len(splittedPerson)-1],
        'occupation': splittedProfession[len(splittedProfession)-1],
    })

df = pd.DataFrame(profession)
#print(df)

unique_persons={}
for a in profession:
    if(a['id'] not in unique_persons):
        unique_persons[a['id']] = [a['occupation']]
    else:
        unique_persons[a['id']].insert(0,a['occupation'])
#print(len(unique_persons))

with open('person_dump.json', 'w') as fp:
    json.dump(unique_persons, fp)