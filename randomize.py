import csv
import random
import json

class Payload(object):
    def __init__(self, j):
        self.__dict__ = json.loads(j)

class User:
    def __init__(self, id, name, country, city, games, groups, events):
        self.id = id
        self.name = name
        self.country = country
        self.city = city
        self.events = events
        self.games = games
        self.groups = groups
        self.friends = []

    def addFriend(self, friend):
        self.friends.append(friend)

    def addEvent(self, event):
        self.events.append(event)


class Event:
    def __init__(self, id, country, city):
        self.id = id
        self.country = country
        self.city = city

games = ['Imago', 'Blood Rage', 'Dust', 'Gipf', 'Pantheon', 'Quadropolis']
groups = ['RONIN', 'ROBINSON ', 'FANATIC', 'BIBLIO']
countries = ['Poland', 'USA', 'Japan', 'Russia']

json_data = open('countries.json').read()
countriesCities = json.loads(json_data)

print countriesCities['Poland']

names = []

with open('names.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile, delimiter='|', dialect=csv.excel_tab)
    for row in reader:
        names.append(row[0])

names.pop(0)

events = []

for i in range(10):
    random.seed()
    name = random.choice(names)
    print name
    country = random.choice(countries)
    print country
    city = random.choice(countriesCities[country])
    print city
    events.append(Event(i, country, city))

users = []

for i in range(100):
    random.seed()
    name = random.choice(names)
    print name
    country = random.choice(countries)
    print country
    city = random.choice(countriesCities[country])
    print city

    defaultProbability = 30

    userGames = []
    for game in games:
        if random.randint(0, 100) < defaultProbability:
            userGames.append(game)
    print userGames

    userGroups = []
    for game in games:
        if random.randint(0, 100) < defaultProbability:
            userGroups.append(game)
    print userGroups

    userEvents = []
    for event in events:
        if event.city == city and random.randint(0, 100) < 50:
            userEvents.append(event)
        elif event.country == country and random.randint(0, 100) < 30:
            userEvents.append(event)
        elif random.randint(0, 100) < 10:
            userEvents.append(event)

    users.append(User(i, name, country, city, userGames, userGroups, userEvents))


for mainUser in users:
    for friendUser in users:
        if mainUser.id != friendUser.id:
            if mainUser.city == friendUser.city and random.randint(0, 100) < 40:
                mainUser.friends.append(friendUser)
            elif mainUser.country == friendUser.country and random.randint(0, 100) < 20:
                mainUser.friends.append(friendUser)
            elif random.randint(0, 100) < 5:
                mainUser.friends.append(friendUser)

print users[4].friends[0].name
