### Contextual feature generator
The code in this directory is used to generator contextual features for ClueWeb12 and the TREC WEB track 2013 & 2014. Because both the ClueWeb12 dataset and index are to large for distribution, it is assumed you already have an index yourself. This code should also be applicable for other index and trec file combinations. Make sure indri and Pyndri are installed before gathering the other prerequisites, instructions can be found [here](https://github.com/cvangysel/pyndri) 

#### Usage
This directory consists of various python files that are able to add queries, documents and contextual features to a .h5 file. This file can then be used for training a learning to rank model. 

- `add_pyndri_index.py`
- `add_pagerank.py`
- `add_anchor_baselines.py`
