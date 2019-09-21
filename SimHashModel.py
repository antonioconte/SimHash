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
    def __init__(self,type="",k='3',t=10,isDataProc = False):
        self.model = None
        self.type = type
        self.tollerance = t
        self.k = k
        config.kGRAM = k
        self.nlp = spacy.load('en_core_web_' + config.size_nlp)
        self.normalizer = text_pipeline.TextPipeline(self.nlp)
        self.isDataProc = isDataProc
        self.pathDataProc = '/home/anto/Scrivania/Tesi/testing/processed_data/'+self.type+'_'+self.k
        if type == 'trigram':
            self.pathmodel = config.path_models + type
        else:
            self.pathmodel = config.path_models + type + '_' + self.k

    def load(self):
        with open(self.pathmodel, 'rb') as handle:
            self.model = pickle.load(handle)

    def train(self):
        print("======= Train: {} [Dataproc: {}, K = {}, Toll: {}] =======".format(self.type,self.isDataProc,self.k, self.tollerance))
        try:
            if type == 'trigram':
                k = 10
                p = iter(Processer(filepath=config.filepath, part=self.type))
                data = []
                for item in tqdm.tqdm(p):
                    data += [{
                        'tag': item[0]['tag'],
                        'data': item[0]['data'][0]
                    }]
                    next(p)
            else:
                if self.isDataProc:

                    with open(self.pathDataProc, 'rb') as handle:
                        data = pickle.load(handle)
                else:
                    p = Processer(filepath=config.filepath, part=self.type)
                    data = p.run()
        except Exception as e:
            print(e)
            return

        time.sleep(1)

        print("======= INDEXING ({}) =======".format(self.type))
        objs = []
        for item in tqdm.tqdm(data):
            objs += [(item['tag'], Simhash(item['data']))]

        start_time = time.time()
        index = SimhashIndex(objs, k=self.tollerance)
        timing_index = "%.2f ms" % ((time.time() - start_time) * 1000)
        print("Indexing time: {}".format(timing_index))

        with open(self.pathmodel, 'wb') as handle:
            pickle.dump(index, handle)

    def predict(self,query,threshold = config.default_threshold,N = config.num_recommendations,Trigram = False):
        if self.model == None:
            raise Exception("Model is not loaded!")

        query = utils.cleanhtml(query)

        if Trigram:
            query, query_norm = self.normalizer.norm_text_trigram(query)
            tokens = query_norm

        else:
            query_norm = self.normalizer.convert(query,divNGram=False)
            tokens = self.normalizer.convert(query)

        start_time = time.time()

        hash_query = Simhash(tokens)

        results = self.model.get_near_dups(hash_query)

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
            res_json = sorted(res_json, key=lambda i: i['lev'], reverse=True)[:config.num_recommendations]
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

