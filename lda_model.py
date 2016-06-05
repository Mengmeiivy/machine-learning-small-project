import logging
import gensim
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


corpus = corpora.MmCorpus('corpus.mm')
dictionary = corpora.Dictionary.load('dict.dict')


# save the models
lda_100 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, update_every=1, chunksize=10000, passes=1)
lda_100.save('lda_100.lda')

lda_5 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)
lda_5.save('lda_5.lda')

lda_20 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, update_every=1, chunksize=10000, passes=1)
lda_20.save('lda_20.lda')

lda_200 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=200, update_every=1, chunksize=10000, passes=1)
lda_200.save('lda_200.lda')

lda_400 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=400, update_every=1, chunksize=10000, passes=1)
lda_400.save('lda_400.lda')
