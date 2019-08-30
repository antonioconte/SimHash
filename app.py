from flask_cors import CORS
from flask import Flask,request
import config
import json
from SimHashModel import SimHashModel

SimHash_f = SimHashModel()
SimHash_p = SimHashModel()
SimHash_s = SimHashModel()
SimHash_t = SimHashModel()

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
	models = []
	msg = "NOT GOOD"
	# load model phrase
	global SimHash_f
	if SimHash_f.model == None:
		SimHash_f.load_model("phrase")
		models.append("Phrase")
	else:
		models.append("Phrase")

	global SimHash_p
	if SimHash_p.model == None:
		SimHash_p.load_model("paragraph")
		models.append("Paragraph")
	else:
		models.append("Paragraph")

	global SimHash_s
	if SimHash_s.model == None:
		SimHash_s.load_model("section")
		models.append("Section")
	else:
		models.append("Section")

	global SimHash_t
	if SimHash_t.model == None:
		SimHash_t.load_model('trigram')
		models.append("TriGram")
	else:
		models.append("TriGram")


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
			'ip': config.ip
		}, indent=4, sort_keys=True),
		status=200,
		mimetype='application/json'
	)
	return response

if __name__ == '__main__':
	app.run('0.0.0.0')
