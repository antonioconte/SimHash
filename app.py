from flask_cors import CORS
from flask import Flask,request
import config
import json
from SimHashModel import SimHashModel

SimHash_f = SimHash_p = SimHash_s = SimHash_t = None
app = Flask(__name__)


CORS(app)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/', methods=['GET'])
def index():
	response = app.response_class(
		response=json.dumps({'data': "Welcome SimHash Entrypoint!"}, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response


@app.route('/query/', methods=['POST'])
def query():
	try:
		query = request.json['data']
		type = request.json['type']
	except:
		return app.response_class(
			response=json.dumps({'error':'query or type is empty'}, indent=4),
			status=505,
			mimetype='application/json'
		)

	try:
		threshold = request.json['threshold']
	except:
		threshold = config.default_threshold

	try:
		maxResults = request.json['max']
	except:
		maxResults = config.num_recommendations


	SimHash_m = None
	T = False
	if type == "Phrase":
		SimHash_m = SimHash_f
	elif type == "Paragraph":
		SimHash_m = SimHash_p
	elif type == "Section":
		SimHash_m = SimHash_s
	elif type == "TriGram":
		T = True
		SimHash_m = SimHash_t

	if SimHash_m == None:
		return app.response_class(
			response=json.dumps({'error': 'type is not valid'}, indent=4),
			status=505,
			mimetype='application/json'
		)

	result = SimHash_m.predict(query,threshold=threshold,N=maxResults,Trigram=T)

	response = app.response_class(
		response=json.dumps(result, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/connect/', methods=['GET'])
def connect():
	k = str(request.args.get('k', default=3, type=int))

	models = []
	msg = "NOT GOOD"
	# load model phrase

	global SimHash_f
	global SimHash_p
	global SimHash_s
	global SimHash_t

	SimHash_f = SimHashModel(type="phrase",k=k)
	SimHash_f.load()
	models.append("Phrase")

	SimHash_p = SimHashModel(type="paragraph", k=k)
	SimHash_p.load()
	models.append("Paragraph")

	SimHash_s = SimHashModel(type="section", k=k)
	SimHash_s.load()
	models.append("Section")

	# SimHash_t = SimHashModel(type="trigram", k=k)
	# SimHash_t.load()
	# models.append("TriGram")


	if len(models) ==  4:
		msg = "All loaded!"
	elif len(models) > 0:
		msg = "loaded"

	response = app.response_class(
		response=json.dumps({
			'data': msg,
			'models': models,
			'path':config.path_models,
			'wordbased': config.wordBased,
			'ip': config.ip,
			'entryname': 'Simhash_k_'+k
		}, indent=4, sort_keys=True),
		status=200,
		mimetype='application/json'
	)
	return response

if __name__ == '__main__':
	app.run('0.0.0.0')
	# port = "1233"
