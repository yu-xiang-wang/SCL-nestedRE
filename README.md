# SCL-nestedRE
These are the source code and data sets for the manuscript: *Nested relation extraction via self-contrastive learning guided by structure and semantic similarity*, submitted to Neural Networks. 
## Requirements
1. python==3.6
2. h5py==3.1.0
3. matplotlib==3.3.4
4. nltk==3.6.7
5. pandas==1.1.5
6. prompt-toolkit==3.0.19
7. scipy==1.5.4
8. torch==torch-1.8.0
## Description
* framework.py: SCL model framework and running file
* config.py: model parameter configuration file
* data_utils.py: data processing code file
* utils.py: indicator calculation code file
* models.py: various relationship extraction model code files
* tiny_models.py: various basic sub-model code files
* fewshot.py: SCL small sample experiment code file
* Details of Experimental Results.xlsx: the details of the experimental results.

## Data Sets
* Data sets in folders data/5-1, data/5-2, data/5-3, data/5-4, and data/5-5 are used for nested relation extraction.
* Data set in folder data/flat-relation is used for nested relation extraction
## Run
python framework.py
