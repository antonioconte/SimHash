size_nlp = "sm"
num_recommendations = 10
default_threshold = 0.3

date_pattern = "(\d{2}|\d{1})(\s{1}|-|/)"+\
"((Jan(uary)?|(Feb(ruary)?|Ma(r(ch)?|y)|Apr(il)?|Jun(e)?|Jul(y)?|Aug(ust)|(Sept|Nov|Dec)(ember)?)|Oct(ober)?)|(\d{1}|\d{2}))"+\
               "(\s{1}|-|/)(\d{4}|(')?\d{2})"

abbr_dict = {
    'CEN': 'European Committee for Standardisation',
    'EEC': 'European Economic Commision',
    'EU': 'European Union',
    'EC': 'European Commission',
    'NATO': 'North Atlantic Treaty Organization',
    'USA': 'United States of America',
    'UK': 'United Kingdom',
    'PGI': 'Principal Global Indicators',
    'PDO': 'Protected designation of origin',
    'EFSA': 'European Food Safety Authority'
}
abbr_expand = "|".join(list(abbr_dict.keys()))
DEBUG = True
wordBased = True

import socket
ip = socket.gethostbyname(socket.gethostname())

if '130.136.4' in ip:
    filepath = './'
    path_models = '/public/antonio_conteduca/model_SimHash/model'
else:
    filepath = '/home/anto/Scrivania/Tesi/dataset_train/'
    path_models = "/home/anto/Scrivania/Tesi/SimHash/model/model"

print(">>>> RUN ON " + ip)