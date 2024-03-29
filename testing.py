import config
from SimHashModel import SimHashModel
import pickle
from tqdm import tqdm
import time
import random
import json

def testing(type='',k='3'):

    if len(type)>0:
        types = [type]
    else:
        types = ['trigram','paragraph', 'section', 'phrase']


    for t in types:

        model = SimHashModel(type=t, k=k)
        model.load()
        print("model load!")
        empty = 0
        T = False

        print("== {} == ".format(t))
        if t == 'trigram':
            T = True
            # per prendere le frasi come spunto per i trigrammi
            t = 'phrase'



        path_test = '/home/anto/Scrivania/Tesi/testing/testing_file/'+t
        with open(path_test, 'rb') as handle:
            queries = pickle.load(handle)



        NUM_TEST = 10
        queries = queries[:NUM_TEST]

        for q in queries:
            res = model.predict(q,Trigram=T)
            if len(res['data']) == 0:
                empty += 1
            print(json.dumps(res, ensure_ascii=False, indent=4))
        time.sleep(0.25)
        print("Empty Result: ", empty)
        print("=====================",t,k, "===========================")
        import gc
        gc.collect()

    exit()

if __name__ == '__main__':
    query = 'regulation shall ent'
    model = SimHashModel(type='trigram')
    model.load()
    res = model.predict(query)
    print(json.dumps(res,indent = 4))
    # testing(type='trigram')

