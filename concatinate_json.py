import os
import json

persons = []

for filename in os.listdir("person_data2/"):
    print("person_data2/"+filename)
    with open("person_data2/"+filename, 'r') as f:
        holder = json.load(f)
    for person in holder:
        persons.append(person)
with open("person_data2/complete_person_dump.json", 'w') as fp:
    json.dump(persons,fp)