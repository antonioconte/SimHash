import re
from SimHash import Simhash, SimhashIndex
from preprocess import process_data
from preprocess import text_pipeline
import spacy
import config
import json
import time
import random
import pickle
from preprocess import metrics
from statistics import mean
import tqdm


class SimHashModel():
    def __init__(self, type, K=20):
        # load model
        try:
            with open('./model/model_' + type + '.pickle', 'rb') as handle:
                self.model = pickle.load(handle)
                self.model.set_k(K)
        except Exception as e:
            self.model = None
            print("Model {} not loaded ({})".format(type,e))

        self.nlp = spacy.load('en_core_web_' + config.size_nlp)
        self.normalizer = text_pipeline.TextPipeline(self.nlp)



    def predict(self,query,threshold = config.default_threshold,N = config.num_recommendations):
        if self.model == None:
            raise Exception("Model is not loaded!")

        # query = cleanhtml(query)

        Trigram = False
        if type == "trigram":
            Trigram = True
            query_norm = ""
            tokens = []
        else:
            query_norm = self.normalizer.convert(query,divNGram=False)
            tokens = self.normalizer.convert(query)

        start_time = time.time()
        hash_query = Simhash(tokens)
        results = self.model.get_near_dups(hash_query)
        # tempo di ricerca
        timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)

        if len(results) == 0:
            res_json = []

        else:
            res_json = [
                metrics.metric(query_norm, doc_retrival, self.normalizer, Trigram=Trigram)
                for doc_retrival in results
            ]
            res_json = [
                res
                for res in sorted(res_json, key=lambda i: i['lev'], reverse=True)
                if float(res['lev']) >= config.default_threshold
            ][:config.num_recommendations]

        # tempo di ricerca + re-ranking
        timing = "%.2f ms" % ((time.time() - start_time) * 1000)


        return {
            'query': query,
            'data': res_json,
            'time': timing,
            'max':N,
            'time_search':timing_search,
            'threshold':threshold
        }

if __name__ == '__main__':
    type = 'paragraph'
    model = SimHashModel(type)
    with open('test_' + type + '.pickle', 'rb') as handle:
        queries = pickle.load(handle)
    random_index = random.randint(1,len(queries))
    query = queries[random_index]
    # query = query[:len(query) - (random.randint(20,len(query))) ]
    # query = query[:int(len(query)/2)]

    print("Predicting....")
    res = model.predict(query)

    print(json.dumps(res,indent=4))