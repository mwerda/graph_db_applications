from neo4j.v1 import GraphDatabase, basic_auth
import copy as cp
import regex
from unidecode import unidecode
import sys

reload(sys)														#necessary to unify the accent marks
sys.setdefaultencoding('utf8')

class Graph:
	#<editor-fold desc="constant strings">
	_db_address = "bolt://localhost"							#bolt is a type of db driver

	_query_del_nodes_and_relations = "MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r"

	_path_all_items = 'u.item'									#paths to specific files
	_path_all_users = 'u.user2'
	_path_all_relations = 'u.data'

	_path_test_items = 'u.itemtest'
	_path_test_users = 'u.usertest'
	_path_test_relations = 'u.datatest'
	#</editor-fold>

	#<editor-fold desc="full datasets loaders">
	def load_all_data(self):									#loads items, users and ratings from movielens
		self.load_items(self._path_all_items)
		self.load_users(self._path_all_users)
		self.load_relations(self._path_all_relations)

	def load_test_data(self):									#loads the small test set
		self.load_items(self._path_test_items)
		self.load_users(self._path_test_users)
		self.load_relations(self._path_test_relations)
	#</editor-fold>

	#<editor-fold desc="file-level loaders">
	def load_items(self, path):
		with open(path) as document:
			for line in document:
				movie = self.cleanse_line(line)					#casting to utf-8 and reducing the alphabet
				movie = movie.replace('\n', '').split('|')		#splits the row by the separator
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

	def load_users(self, path):
		with open(path) as document:
			for line in document:
				movie = self.cleanse_line(line)
				movie = movie.replace('\n', '').split('|')
				query = "CREATE(user:User " \
										 "{id: " + movie[0] + "," \
										 "age: \"" + movie[1] + "\"," \
										 "gender: \"" + movie[2] + "\"," \
										 "occupation: \"" + movie[3] + "\"," \
										 "zip_code: \"" + movie[4] + "\"}" \
										 ")"
				print(query)
				self._session.run(query)

	def load_relations(self, path):
		with open(path) as document:
			for line in document:
				data = self.cleanse_line(line)
				data = data.replace('\n', '').split('\t')
				print(data)

				query = "MATCH (user:User) " \
                    "MATCH (movie:Movie) " \
                    "WHERE user.id = " + data[0] + " " \
                    "AND movie.id = " + data[1] + " " \
                    "CREATE (user)-[:RATED]->(movie)"
				print(query)
				self._session.run(query)
	#</editor-fold>

	#<editor-fold desc="cypher helper queries">
	def wipe(self):
		self._session.run(self._query_del_nodes_and_relations)
	#</editor-fold>

	#<editor-fold desc="helper methods">
	def cleanse_line(self, line):
		clean_line = cp.copy(line)
		clean_line = clean_line.decode('iso-8859-1').encode('utf8')
		clean_line = unidecode(clean_line)
		return clean_line
	#</editor-fold>

	def execute_query(self, query):
		self._session.run(query)

	#<editor-fold desc="constructor, destructor">
	def __init__(self):
		self._driver = GraphDatabase.driver(self._db_address, auth=basic_auth("neo4j", "neo4j"))
		self._session = self._driver.session()

	def __del__(self):
		self._session.close()
	#</editor-fold>


############################################


# graph = Graph()
# graph.wipe()
#
# graph.load_test_data()
#graph.load_all_data()