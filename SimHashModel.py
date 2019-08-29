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
from preprocess import metrics

class SimHashModel():
    def __init__(self,K=20):
        self.model = None
        self.K = K
        self.nlp = spacy.load('en_core_web_' + config.size_nlp)
        self.normalizer = text_pipeline.TextPipeline(self.nlp)

    def load_model(self, type):
        # load model
        try:
            with open(config.path_models + type + '.pickle', 'rb') as handle:
                self.model = pickle.load(handle)
                self.model.set_k(self.K)
        except Exception as e:
            print("Model {} not loaded ({}, path: {})".format(type, e, config.path_models))



    def train(self, type=None, path_model="./model/",path="/home/anto/Scrivania/Tesi/dataset_train/"):
        if type is None:
            print("Type is not declared!!")
            return

        from preprocess import process_data
        try:
            if type == 'trigram':
                limit = 1000
                k = 10
                p = iter(process_data.Processer(path, type))
                data = []
                for item in tqdm.tqdm(p):
                    data += [{
                        'tag': item[0]['tag'],
                        'data': item[0]['data'][0]
                    }]
                    next(p)
            else:
                k = 20
                limit = 100
                p = process_data.Processer(path, type)
                data = p.run()
        except Exception as e:
            print(e)
            return

        time.sleep(1)

        objs = []
        queries = []
        print("======= INDEXING ({}) =======".format(type))
        for item in tqdm.tqdm(data):
            if random.randint(1, limit) == 1:
                queries += [item['tag'].split("]", 1)[1]]
            # print(item['tag'], "_", item['data'], "_", Simhash(item['data']).value)
            objs += [(item['tag'], Simhash(item['data']))]

        start_time = time.time()
        index = SimhashIndex(objs, k=k)
        timing_index = "%.2f ms" % ((time.time() - start_time) * 1000)
        print("Indexing time: {}".format(timing_index))

        # save test_file
        with open('test_' + type + '.pickle', 'wb') as handle:
            pickle.dump(queries, handle)

        # save model
        print("Saving on file: {}".format('test_' + type + '.pickle'))
        with open(path_model+'model_' + type + '.pickle', 'wb') as handle:
            pickle.dump(index, handle)

    def predict(self,query,threshold = config.default_threshold,N = config.num_recommendations,Trigram = False):
        if self.model == None:
            raise Exception("Model is not loaded!")

        query = utils.cleanhtml(query)

        if Trigram:
            query, query_norm = self.normalizer.norm_text_trigram(query)
            print("Query: ",query)
            print("Query Norm: ",query_norm)
            tokens = query_norm
            hash_query_test = Simhash(tokens)
            print("hashValue: ",hash_query_test.value)

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
    model = SimHashModel()

    # ===== PRINT TEST FILE .pickle ========================
    # type = 'trigram'
    # with open('test_' + type + '.pickle', 'rb') as handle:
    #     l = pickle.load(handle)
    # [ print(i) for i in l]
    # print("TOTAL: {}".format(len(l)))
    # exit()

    # ===== TRAIN ==========================================
    # type = 'trigram'
    # config.DEBUG = False
    # model.train(type)
    # exit()


    for t in ['trigram', 'paragraph', 'section', 'phrase'][:1]:
        if t == 'trigram':
            T = True
        type = t
        print("== {} == ".format(t))
        model.load_model(type)
        print("model load!")

        with open('test_' + type + '.pickle', 'rb') as handle:
            queries = pickle.load(handle)

        for i in range(10):

            random_index = random.randint(1,len(queries)-1)
            # random_index = 5
            query = queries[random_index]
            # query = query[:len(query) - (random.randint(20,len(query))) ]
            # query = query[:int(len(query)/2)]
            # states not to allow for any storage _ states allow srage _ 87530828499696193
            # query = "day following its publication"
            print("{}. Predicting....".format(i))
            res = model.predict(query, Trigram=True)

            print(json.dumps(res,indent=4))