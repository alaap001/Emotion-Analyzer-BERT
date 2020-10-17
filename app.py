from flask import Flask , render_template, request

app = Flask(__name__)

from wrangling_scripts.data_wrangle import data_wrangle
import plotly.graph_objs as go
import torch
import plotly , json
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import pickle

data = data_wrangle()
cat_vals = data['category'].value_counts()
x = cat_vals.index
y = cat_vals.values

graph_one = [go.Bar(x = x, y=y, name = 'value_counts')]
figure_Json = json.dumps([{'data':graph_one}],cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html', figure = figure_Json)

# @app.route('/projectOne')
# def projectOne():
# 	return render_template('projectOne.html')

def load_model(weight):
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
														  num_labels = 6,
														  output_attentions = False,
														  output_hidden_states = False)
	print("Loading Weights")
	model.load_state_dict(torch.load(weight,map_location=torch.device('cpu')))
	print("loaded")
	return model

def prepare_text(review_text):

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)

	encoded_review = tokenizer.encode_plus(
	  review_text,
	  max_length=256,
	  truncation=True,
	  add_special_tokens=True,
	  return_token_type_ids=False,
	  pad_to_max_length=True,
	  return_attention_mask=True,
	  return_tensors='pt',
	)

	input_ids = encoded_review['input_ids'].to('cpu')
	attention_mask = encoded_review['attention_mask'].to('cpu')
	
	return (input_ids,attention_mask)

model = load_model('BERT_ft_7.pt')

@app.route('/predict',methods=['POST'])
def predict():
	'''
	For rendering results on HTML GUI
	'''
	
	review_text = [str(x) for x in request.form.values()][0]
	

	input_ids,attention_mask = prepare_text(review_text)

	with torch.no_grad():        
		output = model(input_ids, attention_mask)

	_, prediction = torch.max(output[0], dim=1)

	classes = {2:'angry',
 3:'disgust',
 0:'happy',
 1:'not-relevant',
 4:'sad',
 5:'surprise'}
	
	emotion = classes[int(prediction[0])]

	return render_template('index.html', prediction_text='Emotion of the text {} : {}'.format(review_text,emotion))


app.run(debug=True)
