from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

app = Flask(__name__)
CORS(app)

model_name = "microsoft/DialoGPT-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    global conversation_history
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']
    conversation_history.append(tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt"))
    input_ids = torch.cat(conversation_history, dim=-1)
    output_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    conversation_history.append(tokenizer.encode(response + tokenizer.eos_token, return_tensors="pt"))

    return response

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()