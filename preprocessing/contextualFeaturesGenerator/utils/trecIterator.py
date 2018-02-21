import os.path

class TrecIterator():
	def __init__(self, trec_path):
		self.trec_path = trec_path

	def query_iterator(self):
		location = os.path.join(self.trec_path, "topics.txt")
		with open(location, 'r') as f:
			for line in f:
				idx, query = line.rstrip().split(":", 1)
				yield (idx, query)

	def query_document_iterator(self):
		location = os.path.join(self.trec_path, "qrels.all.txt")

		with open(location, 'r') as f:
			for idx, query in self.query_iterator():
				go = True
				documents = {}
				while go:
					f_pointer = f.tell()
					judgement = f.readline().rstrip().split()

					if len(judgement) == 4:
						if judgement[0] != idx:
							go = False
							f.seek(f_pointer)
						else:
							documents[judgement[2]] = judgement[3]
					else:
						go = False

				yield idx, query, documents

