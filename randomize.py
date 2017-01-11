import csv
import random
import json
from graph import Graph

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
	def __init__(self, id, name, country, city):
		self.id = id
		self.name = name
		self.country = country
		self.city = city


games = ['Imago', 'Blood Rage', 'Dust', 'Gipf', 'Pantheon', 'Quadropolis']
groups = ['RONIN', 'ROBINSON ', 'FANATIC', 'BIBLIO']
countries = ['Poland', 'USA', 'Japan', 'Russia']
users = []
events = []

json_data = open('countries.json').read()
countriesCities = json.loads(json_data)

print countriesCities['Poland']

names = []

with open('names.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile, delimiter='|', dialect=csv.excel_tab)
	for row in reader:
		names.append(row[0])

names.pop(0)

events_names = ['Cinema', 'Party', 'Paintball', 'Kayaking', 'Chess', 'Board games', 'LAN party', 'Carting']
for i in range(10):
	random.seed()
	name = random.choice(names)
	print name
	country = random.choice(countries)
	print country
	city = random.choice(countriesCities[country])
	print city
	name = random.choice(events_names)
	events.append(Event(i, name, country, city))

for i in range(100):
	random.seed()
	name = random.choice(names)
	print name
	country = random.choice(countries)
	print country
	city = random.choice(countriesCities[country])
	print city


	defaultProbability = 30
	largeProbability = 50
	smallProbability = 10

	userGames = []
	for game in games:
		if random.randint(0, 100) < defaultProbability:
			userGames.append(game)
	print userGames

	userGroups = []
	for group in groups:
		if random.randint(0, 100) < defaultProbability:
			userGroups.append(group)
	print userGroups

	userEvents = []
	for event in events:
		if event.city == city and random.randint(0, 100) < largeProbability:
			userEvents.append(event)
		elif event.country == country and random.randint(0, 100) < defaultProbability:
			userEvents.append(event)
		elif random.randint(0, 100) < smallProbability:
			userEvents.append(event)

	users.append(User(i, name, country, city, userGames, userGroups, userEvents))

friendLargeProbability = 40
friendDefaultProbability = 20
friendSmallProbability = 5

for mainUser in users:
	for friendUser in users:
		if mainUser.id != friendUser.id:
			if mainUser.city == friendUser.city and random.randint(0, 100) < friendLargeProbability:
				mainUser.friends.append(friendUser)
			elif mainUser.country == friendUser.country and random.randint(0, 100) < friendDefaultProbability:
				mainUser.friends.append(friendUser)
			elif random.randint(0, 100) < friendSmallProbability:
				mainUser.friends.append(friendUser)


# print users[4].friends[0].name

####### creating a graph
def add_users(users, graph):
	for user in users:
		graph.execute_query(user_to_create_user_query(user))

def add_games(games, graph):
	for game in games:
		graph.execute_query(game_to_create_game_query(game))

def add_groups(groups, graph):
	for group in groups:
		graph.execute_query(group_to_create_group_query(group))

def add_countries(countries, graph):
	for country in countries:
		graph.execute_query(country_to_create_country_query(country))

def add_events(events, graph):
	for event in events:
		graph.execute_query(event_to_create_event_query(event))

def connect_events_with_places(events, graph):
	for event in events:
		query1 = "MATCH (event:Event) " \
                    "MATCH (country:Country) " \
                    "WHERE event.country = country.name " \
                    "CREATE (event)-[:HAPPENS_IN]->(country)"

		query2 = "MATCH (event:Event) " \
				 "MATCH (city:City) " \
				 "WHERE event.city = city.name " \
				 "CREATE (event)-[:HAPPENS_IN]->(city)"

		graph.execute_query(query1)
		graph.execute_query(query2)

def add_cities(countries_cities, graph):
	for key in countries_cities:
		for element in countries_cities[key]:
			graph.execute_query("CREATE(city:City " \
		   "{name: \"" + element + "\" })")


def create_connections_of_users(users, graph):
	match_country = "MATCH (user:User) " \
					"MATCH (country:Country) " \
					"WHERE user.city = country.name " \
					"CREATE (user)-[:LIVES_IN]->(country)"
	graph.execute_query(match_country)

	match_city = "MATCH (user:User) " \
				 "MATCH (city:City) " \
				 "WHERE user.city = city.name " \
				 "CREATE (user)-[:LIVES_IN]->(city)"
	graph.execute_query(match_city)

	for user in users:
		for event in user.events:
			graph.execute_query("MATCH (user:USER) " \
				 "MATCH (event:Event) " \
				 "WHERE user.id = " + str(user.id) + " " \
				 "AND event.id = " + str(event.id)  + " " \
				 "CREATE (user)-[:TAKES_PART_IN]->(event)")

		for game in user.games:
			graph.execute_query("MATCH (user:USER) " \
				 "MATCH (game:Game) " \
				 "WHERE user.id = " + str(user.id) + " "\
				 "AND game.title = \"" + str(game.title)  + "\" " + \
				 "CREATE (user)-[:PLAYS]->(game)")

		for group in user.groups:
			graph.execute_query("MATCH (user:USER) " \
				 "MATCH (group:Group) " \
				 "WHERE user.id = " + str(user.id) + " "\
				 "AND group.title = \"" + str(group.title)  + "\" " \
				 "CREATE (user)-[:BELONGS_TO]->(group)")



def user_to_create_user_query(user):
	return "CREATE(user:User " \
		   "{name: \"" + user.name + "\", id: \"" + str(user.id) + "\"})"

def game_to_create_game_query(game):
	return "CREATE(game:Game " \
		   "{title: \"" + game + "\"})"


def group_to_create_group_query(group):
	return "CREATE(group:Group " \
		   "{title: \"" + group + "\"})"


def country_to_create_country_query(country):
	return "CREATE(country:Country " \
		   "{name: \"" + country + "\"})"

def event_to_create_event_query(event):
	return "CREATE(event:Event " \
		   "{name: \"" + event.name + \
		   "\", country: \"" + event.country + \
		   "\", id: " + str(event.id) + \
		   ", city: \"" + event.city + "\"})"

graph = Graph()
graph.wipe()

print()
add_users(users, graph)
add_games(games, graph)
add_groups(groups, graph)
add_countries(countries, graph)
add_cities(countriesCities, graph)
add_events(events, graph)

connect_events_with_places(events, graph)
create_connections_of_users(users, graph)

print()
