import logging
import gensim
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load the model 
model = gensim.models.ldamodel.LdaModel.load('lda_5.lda')

# print all topics in the model 
model.print_topics(-1)