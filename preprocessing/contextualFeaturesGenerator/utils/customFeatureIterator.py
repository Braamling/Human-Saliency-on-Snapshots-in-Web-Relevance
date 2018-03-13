import os.path

class CustomFeatureIterator():
	def __init__(self, path):
		self.path = path

	"""

	"""
	def line_iterator(self):
		with open(self.path, 'r') as f:
			for line in f:
				yield line.rstrip().split(" ")

	def feature_iterator(self):
		for features in self.line_iterator():
			query_id = features[0] 
			doc_id = features[1]
			rel_score = features[2]
			features = features[3:]
			yield query_id, doc_id, rel_score, features
