import os.path

class LETORIterator():
	def __init__(self, path):
		self.path = path

	"""

	"""
	def file_iterator(self):
		with open(self.path, 'r') as f:
			for line in f:
				yield line

	def feature_iterator(self):
		for line in self.file_iterator():
			scores, notes = line.rstrip().split(" #")
			scores = scores.split(" ")
			rel_score = scores[0] 
			query_id = scores[1].split(":")[1]
			features = [x.split(":")[1] for x in scores[2:]]

			#docid = clueweb12-0207wb-93-10480
			#docid = GX000-00-0000000 inc = 1 prob = 0.0214125
			# temporary solution, do more stable parsing for docid
			doc_id = notes.replace(" = ", "=").split(" ")[0].split("=")[1]
			yield query_id, doc_id, rel_score, features

	def line_iterator(self):
		for line in self.file_iterator():
			scores, notes = line.rstrip().split(" #")
			scores = scores.split(" ")
			query_id = scores[1].split(":")[1]
			yield query_id, line