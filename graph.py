from neo4j.v1 import GraphDatabase, basic_auth
import copy as cp
import regex
from unidecode import unidecode
import sys

reload(sys)
sys.setdefaultencoding('utf8')

class Graph:
	_db_address = "bolt://localhost"
	_query_del_nodes_and_relations = "MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r"

	def load_movies(self):
		with open('u.item') as document:
			for line in document:
				movie = self.cleanse_line(line)
				movie = movie.replace('\n', '').split('|')
				query = "CREATE(movie:Movie " \
										 "{id: " + movie[0] + "," \
										 "movie_title: \"" + movie[1] + "\"," \
										 "release_date: \"" + movie[2] + "\"," \
										 "video_release_date: \"" + movie[3] + "\"," \
										 "url: \"" + movie[4] + "\"," \
										 "unknown: \"" + movie[5] + "\"," \
										 "action: \"" + movie[6] + "\"," \
										 "adventure: \"" + movie[7] + "\"," \
										 "animation: \"" + movie[8] + "\"," \
										 "children: \"" + movie[9] + "\"," \
										 "comedy: \"" + movie[10] + "\"," \
										 "crime: \"" + movie[11] + "\"," \
										 "documentary: \"" + movie[12] + "\"," \
										 "drama: \"" + movie[13] + "\"," \
										 "fantasy: \"" + movie[14] + "\"," \
										 "film_noir: \"" + movie[15] + "\"," \
										 "horror: \"" + movie[16] + "\"," \
										 "musical: \"" + movie[17] + "\"," \
										 "mystery: \"" + movie[18] + "\"," \
										 "romance: \"" + movie[19] + "\"," \
										 "sci_fi: \"" + movie[20] + "\"," \
										 "thriller: \"" + movie[21] + "\"," \
										 "war: \"" + movie[22] + "\"," \
										 "western: \"" + movie[23] + "\"}" \
										 ")"
				print(query)
				self._session.run(query)

	def load_users(self):
		with open('u.user') as document:
			for line in document:
				movie = self.cleanse_line(line)
				movie = movie.replace('\n', '').split('|')
				query = "CREATE(movie:Movie " \
										 "{id: " + movie[0] + "," \
										 "age: \"" + movie[1] + "\"," \
										 "gender: \"" + movie[2] + "\"," \
										 "occupation: \"" + movie[3] + "\"," \
										 "zip_code: \"" + movie[4] + "\"}" \
										 ")"
				print(query)
				self._session.run(query)

	def load_relations(self):
		with open('u.data') as document:
			for line in document:
				data = self.cleanse_line(line)
				data = data.replace('\n', '').split('\t')
				#print(data)

				query = "MATCH (user:User) " \
                    "MATCH (movie:Movie) " \
                    "WHERE user.id = \"" + data[0] + "\" " \
                    "AND movie.id = \"" + data[1] + "\" " \
                    "CREATE (user)-[:RATED]->(movie)"
				#print(query)
				self._session.run(query)

	def cleanse_line(self, line):
		clean_line = cp.copy(line)
		clean_line = clean_line.decode('iso-8859-1').encode('utf8')
		clean_line = unidecode(clean_line)
		return clean_line

	def wipe(self):
		self._session.run(self._query_del_nodes_and_relations)

	def __init__(self):
		self._driver = GraphDatabase.driver(self._db_address, auth=basic_auth("neo4j", "neo4j"))
		self._session = self._driver.session()

	def __del__(self):
		self._session.close()

graph = Graph()
graph.wipe()
# graph.load_movies()
# graph.load_users()
# graph.load_relations()