import logging
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# save the corpus 
corpus = corpora.ucicorpus.UciCorpus('docword.nytimes.txt', 'vocab.nytimes.txt')
corpora.MmCorpus.serialize('corpus.mm', corpus)

# save the dictionary 
dictionary = corpus.create_dictionary()
dictionary.save('dict.dict')


