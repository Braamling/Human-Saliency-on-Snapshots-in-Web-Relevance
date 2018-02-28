# ClueWeb12 contextual features extractor
This directory contains Spark scala code for retrieving the TF, IDF and average document length for the ClueWeb12 collection. The final output is a set of Spark parquet DataFrame containing the TF and IDF for each words in the document as a SparseVector. 

TODO create a python script that can convert the parquet DataFrame to a Pandas DataFrame.