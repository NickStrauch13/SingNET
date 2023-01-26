import pandas as pd
import json
import sqlite3
import os
#This code may break if you change it Go to DBQueries to do your statistics.

allJsonsPreliminary=os.listdir('data/')
allJsons=[]
print(allJsonsPreliminary)
for i in range(len(allJsonsPreliminary)):
    if(allJsonsPreliminary[i].__contains__(".json")):
        allJsons.append(allJsonsPreliminary[i])
print(allJsons)
with open("data/Abbott's_Babbler.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
connection = sqlite3.connect("birdrecordingdb.sqlite")
cursor = connection.cursor()
cursor.execute("Create Table if not exists BirdRecordings (id Text, gen Text, sp Text,ssp Text, gr Text,"
               " en Text,rec Text, cnt Text, loc Text, lat Text, lng Text,"
               " alt Text,type Text, sex Text, stage Text, method Text,"
               " url Text,file Text, filename Text,"
               " lic Text,q Text, length Text, time Text, date Text, "
               "uploaded Text, rmk Text, birdseen Text,"
               " animalseen Text,playbackused Text, temp Text,"
               " regnr Text, auto Text, dvc Text,mic Text, smp Text)")


columns = ['id', 'gen', 'sp','ssp', 'group', 'en',
           'rec', 'cnt', 'loc', 'lat', 'lng', 'alt',
           'type', 'sex', 'stage', 'method', 'url',
           'file', 'file-name',  'lic',
           'q', 'length', 'time', 'date', 'uploaded',
            'rmk', 'bird-seen', 'animal-seen',
           'playback-used', 'temp', 'regnr', 'auto', 'dvc',
           'mic', 'smp']
for i in range(len(allJsons)):
    traffic = json.load(open("data/"+allJsons[i]))
    for row in traffic:
        keys= tuple(row[c] for c in columns)
        cursor.execute('insert into BirdRecordings values(?,?,?,'
                       '?,?,?,?,?,?,?,?,?,?,?,?,'
                       '?,?,?,?,?,?,?,?,?,?,?,?,'
                       '?,?,?,?,?,?,?,?)', keys)
    print(f'{row["en"]} data inserted Successfully')
connection.commit()
connection.close()


