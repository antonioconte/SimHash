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

type ="section"

k = [20]
def testing():
    F = 64
    str1 = "the european parliament and the council of the european union, having regard to the treaty establishing the european community, and in particular (1) thereof, and thereof in relation to articles 17, 18 and 19 of this directive, having regard to the proposal from the commission, having regard to the opinion of the european economic and social committee, having regard to the opinion of the committee of the regions, acting in accordance with the procedure laid down in of the treaty, whereas: (1) the control of european energy consumption and the increased use of energy from renewable sources, together with energy savings and increased energy efficiency, constitute important parts of the package of measures needed to reduce greenhouse gas emissions and comply with the kyoto protocol to the united nations framework convention on climate change, and with further community and international greenhouse gas emission reduction commitments beyond 2012. those factors also have an important part to play in promoting the security of energy supply, promoting technological development and innovation and providing opportunities for employment and regional development, especially in rural and isolated areas."
    str2 = "the european parliament and the council of the european union, having regard to the treaty establishing the european community, and in particular (1) thereof, and thereof in relation to articles 17, 18 and 19 of this directive, having regard to the proposal from the commission, having regard to the opinion of the european economic and social committee, having regard to the opinion of the committee of the regions, acting in accordance with the procedure laid down in of the treaty, whereas: (1) the control of european energy consumption and the increased use of energy from renewable sources, together with energy savings and increased energy efficiency, constitute important parts of the package of measures needed to reduce greenhouse gas emissions and comply with the kyoto protocol to the united nations framework convention on climate change, and with further community and international greenhouse gas emission reduction commitments beyond 2012. those factors also have an important part to play in promoting the security of energy supply, promoting technological development and innovation and providing opportunities for employment and regional development, especially in rural and isolated areas."

    nlp = spacy.load('en_core_web_' + config.size_nlp)
    pip = text_pipeline.TextPipeline(nlp)

    norm1 = pip.convert(str1)
    norm1_union = pip.convert(str1)
    hash1 = Simhash(norm1,f=F)


    norm2 = pip.convert(str2)
    norm2_union = pip.convert(str2)
    hash2 = Simhash(norm2,f=F)


    print("1 str:",str1)
    print("1 normalizer:",norm1_union)
    print("1 div:",norm1)
    print("1 hash:",hash1.value)
    print()
    print("2 str:",str2)
    print("2 normalizer:",norm2_union)
    print("2 norm:",norm2)
    print("2 hash:",hash2.value)

    print("\ndiff:",hash1.distance(hash2))
    exit(1)
# testing()

train = True
config.DEBUG = False

if train:
    p = process_data.Processer('/home/anto/Scrivania/Tesi/dataset_train/', type)
    data = p.run()
    objs = []
    queries = []
    for item in data:
        if random.randint(1,100) == 1:
            queries += [item['tag'].split("]",1)[1]]
        objs += [ (item['tag'],Simhash(item['data']))]


    start_time = time.time()
    index = SimhashIndex(objs,k=20)
    timing_index = "%.2f ms" % ((time.time() - start_time) * 1000)
    print("Indexing time: {}".format(timing_index))

    # save test_file
    with open('test_'+type+'.pickle', 'wb') as handle:
        pickle.dump(queries, handle)

    # save model
    with open('./model/model_'+type+'.pickle', 'wb') as handle:
        pickle.dump(index, handle)

else:

    with open('test_'+type+'.pickle', 'rb') as handle:
        queries = pickle.load(handle)

    # load model
    with open('./model/model_'+type+'.pickle', 'rb') as handle:
        index = pickle.load(handle)

    nlp = spacy.load('en_core_web_' + config.size_nlp)
    pip = text_pipeline.TextPipeline(nlp)



    for K in k:
        num_queries = len(queries)
        no_results = 0
        mean_metrics = []
        mean_time = []
        index.set_k(K)
        for query in tqdm.tqdm(queries):

            query_norm = pip.convert(query,divNGram=False)

            start_time = time.time()
            hash_query = Simhash(pip.convert(query))
            results = index.get_near_dups(hash_query)

            if len(results) == 0:
                num_queries -= 1
                no_results += 1
                res_json = []

            else:
                num_queries -= 1
                res_json = [
                    metrics.metric(query_norm, doc_retrival, pip, Trigram=False)
                    for doc_retrival in results
                ]
                res_json = [
                    res
                    for res in sorted(res_json, key=lambda i: i['lev'], reverse=True)
                    if float(res['lev']) >= config.threshold
                ][:config.num_results]

            timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)
            values = [float(item['lev']) for item in res_json]
            if len(values) > 0:
                mean_metrics += [mean(values)]
            else:
                mean_metrics += [0]

            mean_time += [float(timing_search.split(" ")[0])]
            result = {
                'query': query,
                'query_time':timing_search,
                'data': res_json
            }
            # print(json.dumps(result,indent=4))
            # print()
        time.sleep(0.25)
        assert num_queries == 0, "Error"
        print("\n============== K = {} ==========".format(K))
        print("NoResult: {}".format(no_results))
        print("Avg Time: {} ms".format(round(mean(mean_time),2)))
        print("Avg Metric: {} - {}".format(round(sum(mean_metrics)/len(mean_metrics),2),mean_metrics))
        print("================================")
        time.sleep(1)
