import pyndri
import operator
import re

"""
This class acts as an interface to the pyndri query environnmets.
It makes use of the available Environments to score a list of ext_document_ids
together with a query for each available query enviroment.
"""
class PyndriScorer():
    def __init__(self, index_path):
        self.index = pyndri.Index(index_path)
        self.tfidf_query_env = pyndri.TFIDFQueryEnvironment(self.index)
        self.bm25_query_env = pyndri.OkapiQueryEnvironment(self.index)
        self.lm_query_env = pyndri.QueryEnvironment(self.index, rules=('method:dirichlet,mu:5000',))

    def bm_scores(self, documents, query):
        for i in self.scores(documents, query, self.bm25_query_env):
            yield i
        
    def tfidf_scores(self, documents, query):
        for i in self.scores(documents, query, self.tfidf_query_env):
            yield i   
             
    def lm_scores(self, documents, query):
        for i in self.scores(documents, query, self.lm_query_env):
            yield i

    def scores(self, documents, query, env):
        # Filter the query from all non a-z, 0-9 characters.
        query = self.filter_query(query)

        # Limit the search to the judged documents
        document_set = map(operator.itemgetter(1), self.index.document_ids(documents))

        # Get the results for the provided query.
        results = env.query(query, document_set=document_set, results_requested=len(documents))
        
        documents = set(documents)
        scored_docs = set()

        # return all the scored documents
        for int_doc_id, score in results:
            ext_doc_id = self.index.document(int_doc_id)[0]
            scored_docs.add(ext_doc_id)
            yield ext_doc_id, score

        # return all non scored documents as score 0
        # TODO is this because they are not in the index or because they are scored 0.
        # for ext_doc_id in documents - scored_docs:
        #     yield ext_doc_id, 0

    def filter_query(self, query):
        return re.sub('[^a-zA-Z0-9 .-]','', query)