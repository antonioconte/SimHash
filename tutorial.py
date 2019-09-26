import re
import pickle
import json
from preprocess.metrics import metric
type = 'phrase'
k = '1'

SGN_L = 128
TOLLERANCE = 30
with open('/home/anto/Scrivania/Tesi/testing/processed_data/'+type+'_'+k, 'rb') as f:
    data = pickle.load(f)

# data = data[:400]


from SimHash import Simhash, SimhashIndex
from tqdm import tqdm

objs = [(d['tag'], Simhash(d['data'],f=SGN_L)) for d in tqdm(data)]
index_lsh = SimhashIndex(objs, f=SGN_L, k=TOLLERANCE)

print("END INDEX")

# print(index_lsh.bucket_size())

import time

for id in [0,1,2,3,4]:

    query = data[id]['tag'].split("]",1)[1][:-50]
    print("Query : >",query)

    s1 = data[id]['data']
    print("NORMALIZED: ",s1)

    start_time = time.time()
    s1_hash = Simhash(s1,f=SGN_L)
    results = index_lsh.get_near_dups(s1_hash)
    timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)

    if len(results) == 0:
        res_json = []

    else:
        print('\n', timing_search)
        print('#Res: ', len(results))
        import spacy
        from preprocess.text_pipeline import TextPipeline
        nlp = spacy.load('en_core_web_sm')
        normalizer = TextPipeline(nlp)
        query_norm = normalizer.convert(query,divNGram=False)
        res_json = []
        for doc_retrival in results:
            item = metric(query_norm, doc_retrival, normalizer, Trigram=False)
            if float(item['lev']) >= 0.3:
                res_json += [item]
        # ====== RE-RANKING =========================================================
        res_json = sorted(res_json, key=lambda i: i['lev'], reverse=True)[:10]

    # tempo di ricerca + re-ranking
    timing = "%.2f ms" % ((time.time() - start_time) * 1000)

    print(json.dumps({
        "data": res_json,
        "time": timing,
        "time query": timing_search
    }, indent=4))
    print("==================================================")