
import requests
import pandas as pd

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
SELECT ?person ?item ?itemLabelOcc (lang(?itemLabel) as ?lang)
WHERE 
{
    ?person wdt:P31 wd:Q5 .
    ?person p:P106 ?occupation .
    ?occupation ps:P106 ?item .
    ?item rdfs:label ?itemLabelOcc .
    FILTER (lang(?itemLabelOcc) = "en") .
}
LIMIT 100000 '''


url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
#had to use the flag 'strict: false' in order to escape the \n and \r
data = requests.get(url, params={'query': prefixes + english_query, 'format': 'json', 'strict': 'false'}).json()
# print(data)

profession = []
for item in data['results']['bindings']:
    profession.append({
        'id': item.get('person', {}).get('value'),
        'occupation': item.get('itemLabelOcc', {}).get('value'),
    })

df = pd.DataFrame(profession)
#print(df)

#print("This is a test::::::::")
unique_persons={}
#print(len(profession))

for a in profession:
    if(a['id'] not in unique_persons):
        unique_persons[a['id']] = [a['occupation']]
    else:
        unique_persons[a['id']].insert(0,a['occupation'])
print(unique_persons)