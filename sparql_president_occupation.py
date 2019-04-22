"""
Modified from http://ramiro.org/notebook/us-presidents-causes-of-death/
"""
import requests
import pandas as pd
from collections import OrderedDict

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 80
pd.options.display.width = 200

query = '''PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?president ?cause ?dob ?dod 
WHERE 
{
    ?pid wdt:P39 wd:Q11696 .
    ?pid wdt:P509 ?cid .
    ?pid wdt:P569 ?dob .
    ?pid wdt:P570 ?dod .

    OPTIONAL {
        ?pid rdfs:label ?president filter (lang(?president) = "fr") .
    }
    OPTIONAL {
        ?cid rdfs:label ?cause filter (lang(?cause) = "sv") .
    }
}
'''

url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
data = requests.get(url, params={'query': query, 'format': 'json'}).json()


presidents = []
for item in data['results']['bindings']:
    presidents.append(OrderedDict({
        'name': item['president']['value'],
        'date of birth': item['dob']['value'],
        'date of death': item['dod']['value'],
        'cause of death': item.get('cause', {}).get('value')}))

df = pd.DataFrame(presidents)
print(len(df))
print(df)