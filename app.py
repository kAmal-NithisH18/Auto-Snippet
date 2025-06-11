from flask import Flask, request, jsonify, render_template
from ngram_model import tokenize_python_code, ngram_model
from RNN_BasedModelLoading import generate_codeLSTM, generate_codeGRU, modelLSTM, modelGRU
from TransformerLoading import modelTransformer, generate_Transformer
from GPT import fintuned_GPT
from codeT5 import codeT5_generate
from chatBot import chatResponse
from SubProcessRun import run_SubProcesscode
import requests
import json
import uuid




app = Flask(__name__)

# Dictionary to store available models
suggestion_models = {
    "ngram": ngram_model,
    "lstm": modelLSTM,
    "gru": modelGRU,
    "transformer": modelTransformer,
    "gpt" : fintuned_GPT,
    "codet5" : codeT5_generate

}

# Chat history to maintain context for each session
chat_sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    text = data.get("text", "")
    model_name = data.get("model", "ngram")  # Default to ngram if not specified
    
    # Get the selected model
    model = suggestion_models.get(model_name)
    print("called -app.y")
    
    suggestion = ""
    if model_name == "ngram":
        text = tokenize_python_code(text)
        suggestion = model.generate(text[-4:]) if len(text) >= 2 else ""
    
    elif model_name == "codet5":
        suggestion = codeT5_generate(text)
    elif model_name == "lstm":
        suggestion = generate_codeLSTM(model, text)
    elif model_name == "gru":
        suggestion = generate_codeGRU(model, text)
    elif model_name == "transformer":
        suggestion = generate_Transformer(model, text, max_length=50)
    elif model_name == "gpt":
        suggestion = fintuned_GPT(text)
    else:
        suggestion = " " #Else handle
    
    
    return jsonify({"suggestion": suggestion})

@app.route('/run_code', methods=['POST'])
def run_code():
    code = request.json.get('code', " ")
    
    # Placeholder for code execution
    output , status = run_SubProcesscode(code)
    if status:
        return jsonify({
                'output': output,
                'error': '',
                'success': True})
    else:
        return jsonify({
            'output': '',
            'error': str(output),
            'success': False
        })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    #print(message)
    code = data.get('code', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    # Initialize session if it doesn't exist
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Add user message to history
    chat_sessions[session_id].append({
        "role": "user",
        "content": message
    })
    
    try:

        response_text = chatResponse(message)
        print(response_text)
        
        # Add bot response to history
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_text
        })
        
        # Maintain a reasonable history size (optional)
        if len(chat_sessions[session_id]) > 20:
            chat_sessions[session_id] = chat_sessions[session_id][-20:]
        
        return jsonify({
            "response": response_text,
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({
            "response": f"Sorry, I encountered an error: {str(e)}",
            "session_id": session_id
        })
    
if __name__ == '__main__':
    app.run(debug=False)