from SimHash import Simhash, SimhashIndex
from preprocess import text_pipeline
from preprocess import utils
import spacy
import config
import json
import time
import random
import pickle
import tqdm
from preprocess.process_data import Processer
from preprocess import metrics

class SimHashModel():
    def __init__(self,type="",k='3',T=30,sign=128):
        self.model = None
        self.type = type
        self.tollerance = T
        self.k = k
        config.kGRAM = k
        self.nlp = spacy.load('en_core_web_' + config.size_nlp)
        self.normalizer = text_pipeline.TextPipeline(self.nlp)

        if type == 'trigram':
            self.pathmodel = config.path_models + type
            self.sign = 64
            self.tollerance = 5
            self.k = '3'

        else:
            self.tollerance = T
            self.k = k
            self.sign = sign
            self.pathmodel = config.path_models + type + '_' + self.k

        self.pathDataProc = config.pathDataProc.format(self.type,self.k)

    def load(self):
        with open(self.pathmodel, 'rb') as handle:
            self.model = pickle.load(handle)

    def train(self):
        print("======= Train: {} [K = {}, Toll: {}, Sign = {}] =======".format(self.type,self.k, self.tollerance, self.sign))
        with open(self.pathDataProc, 'rb') as handle:
            data = pickle.load(handle)

        objs = []
        for item in tqdm.tqdm(data):
            tokens = item['data']


            if self.type == 'trigram':
                tokens = self.normalizer.generate_ngrams_char(item['data'][0])

            objs += [(item['tag'], Simhash(tokens ,f=self.sign))]

        start_time = time.time()
        index = SimhashIndex(objs,f=self.sign, k=self.tollerance)
        timing_index = "%.2f ms" % ((time.time() - start_time) * 1000)
        print("Indexing time: {}".format(timing_index))

        with open(self.pathmodel, 'wb') as handle:
            pickle.dump(index, handle)

    def predict(self,query,threshold = config.default_threshold,N = config.num_recommendations,Trigram = False):

        query = utils.cleanhtml(query)

        if self.type == 'trigram':
            query, query_norm = self.normalizer.get_last_trigram(query)
            if query_norm == None:
                return {'query': query,
                        'data': [],
                        'time': '0 ms',
                        'max': N,
                        'time_search': '0 ms',
                        'threshold': threshold}
            else:
                Trigram = True
                tokens = self.normalizer.generate_ngrams_char(query_norm)

        else:
            Trigram = False
            query_norm = self.normalizer.convert(query, False)
            tokens = self.normalizer.convert(query)


        start_time = time.time()

        hash_query = Simhash(tokens,f=self.sign)

        results = self.model.get_near_dups(hash_query,n=config.num_recommendations)

        timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)

        if len(results) == 0:
            res_json = []

        else:
            res_json = []
            for doc_retrival in results:
                item = metrics.metric(query_norm, doc_retrival, self.normalizer, Trigram=Trigram)
                if float(item['lev']) >= threshold:
                    res_json += [item]
            # ====== RE-RANKING =========================================================
            res_json = sorted(res_json, key=lambda i: i['lev'], reverse=True)
        #
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

