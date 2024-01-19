from datasets import load_dataset
import spacy
import nltk
import tqdm

import gensim.downloader as api
from embeddings import get_GloVe_embeds

from transformers import BertModel, BertTokenizer 
from transformers import T5Tokenizer, T5Model 
from sentence_transformers import SentenceTransformer

import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')


# preparing data
# --------------
print("\nPreparing data...\n")

# needed
load_dataset('mt0rm0/movie_descriptors_small', split='train')

# optional
# load_dataset('mt0rm0/movie_descriptors', split='train')

print("Data ready\n\n-----\n")

# preparing normalizing tools
# ---------------------------
print("\nPreparing normalizing tools...\n")

# needed
spacy.cli.download(f'en_core_web_sm')
nltk.download('punkt')
nltk.data.find('corpora/stopwords')

print("Normalizing tools ready\n\n-----\n")

# preparing static embeddings
# ---------------------------
print("\nPreparing static embeddings...\n")

# needed
# word2Vec embeddings
api.load('word2vec-google-news-300')

# GloVe embeddings
get_GloVe_embeds()

# optional
# (Skipgram) Word2Vec model
# Model GoogleNews-vectors-negative300 needs to be downloaded manually and extracted in the folder data/external.
# Link for the download:
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g


# FastText embeddings
# from src.embeddings import get_FastText_embeds
# get_FastText_embeds()

# doc2vec embeddings
# Models apnews_dbow.tgz and enwiki_dbow.tgz need to be downloaded
# manually and extracted in the folder data/external.
# Link for the download:
# https://unimelbcloud-my.sharepoint.com/personal/jeyhan_lau_unimelb_edu_au/_layouts/15/onedrive.aspx
# Folder: public-datasets-models/2016-doc2vec

print("Static embeddings ready\n\n-----\n")

# preparing contextual embeddings
# -------------------------------
print("\nPreparing contextual embeddings...\n")

# needed
# Bert Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
model = BertModel.from_pretrained("bert-base-uncased")

# T5 Model
T5Tokenizer.from_pretrained('t5-base') 
T5Model.from_pretrained('t5-base') 

# from SentenceTransformers
SentenceTransformer('all-MiniLM-L6-v2')

# optional 
# RoBERTa Model
# from transformers import RobertaTokenizer, RobertaModel 
# RobertaTokenizer.from_pretrained('roberta-base') 
# RobertaModel.from_pretrained('roberta-base') 

# XLNet:
# from transformers import XLNetTokenizer, XLNetModel 
# XLNetTokenizer.from_pretrained('xlnet-base-cased') 
# XLNetModel.from_pretrained('xlnet-base-cased') 

# Electra:
# from transformers import ElectraTokenizer, ElectraModel 
# ElectraTokenizer.from_pretrained('google/electra-small-discriminator') 
# ElectraModel.from_pretrained('google/electra-small-discriminator') 

# from SentenceTransformers
# SentenceTransformer('all-MiniLM-L12-v2')
# SentenceTransformer('all-mpnet-base-v2')

print("Contextual embeddings ready\n\n-----\n")