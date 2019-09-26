import json
import random
type = ['paragraph', 'section', 'phrase']

# for t in type:
#     count = 0
#     path = "/home/anto/Scrivania/Tesi/dataset_train/total_{}.json".format(t)
#     with open(path) as json_file:
#         data = json.load(json_file)
#     data = data['data']
#     docs = data.keys()
#     for d in docs:
#         count += len(data[d])
#     print(t, count)
#
# exit(1)

res = {
    'total': {},
    'paragraph': [],
    'section':[],
    'phrase':[]
}

for t in type:
    # path = "/home/anto/Scrivania/Tesi/dataset_train/total_{}.json".format(t)
    path = '/home/anto/Scrivania/Tesi/dataset_test/test_total_{}.json'.format(t)

    with open(path) as json_file:
        data = json.load(json_file)

    data = data['data']
    docs = data.keys()
    for d in docs:
        for r in data[d]:
            if random.randint(1, 4) == 1:
                if len(r) > 10:
                    res[t] += [r]

res['total']['paragraph'] = len(res['paragraph'])
res['total']['section'] = len(res['section'])
res['total']['phrase'] = len(res['phrase'])
print('Paragrafi:',len(res['paragraph']))
print('Sezioni: ',len(res['section']))
print('Frasi: ',len(res['phrase']))

f = open("testing.json",'w')
f.write(json.dumps(res, indent=4, sort_keys=True,ensure_ascii=False))
f.close()

