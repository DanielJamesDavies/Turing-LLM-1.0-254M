import os
from flask import Blueprint, request, jsonify
from app.llm import LLM

generate_bp = Blueprint('generate_bp', __name__)

@generate_bp.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    
    if "prompt" not in data:
        return jsonify({"error": "Request must contain a prompt"}), 400
    prompt = data["prompt"]
    
    messages = [ {"role": "user", "content": prompt } ]
    
    max_length = 8000
    if "max_length" in data:
        max_length = data["max_length"]
        
    llm = LLM()
    
    skip_special_tokens = True    
    if "type" in data and data["type"] == "set":
        skip_special_tokens = False
    if "skip_special_tokens" in data:
        skip_special_tokens = data["skip_special_tokens"]
        
    generated_text, _ = llm.generate(messages, max_length, skip_special_tokens)
    
    if "type" in data:
        if data["type"] == "set":
            generated_text = process_set(data, generated_text)
    
    return jsonify({ 'message': 'Success', 'generated_text': generated_text })

def process_set(data, generated_text):
    decoded_list = [e.strip().rstrip(".") for e in generated_text.split(", ")]
    
    if decoded_list[-1][-7:] != "<|end|>":
        decoded_list.pop()
    else:
        decoded_list[-1] = decoded_list[-1][:-7]
    
    if "file" in data and data["file"][-4:] == ".txt":
        os.makedirs("./dataset", exist_ok=True)
        
        if data["file_operation"] == "append":
            file_path = data["file"]
            try:
                file_contents = open(f'./dataset/{file_path}', 'r', encoding='utf-8').read()
                decoded_list = file_contents.split("\n") + decoded_list
            except:
                print("", end="")
        
        decoded_list = "\n".join(list(dict.fromkeys(decoded_list)))
        
        save_txt(file_path, decoded_list)
    else:
        decoded_list = "\n".join(list(dict.fromkeys(decoded_list)))
        
    return decoded_list
    
def save_txt(file_path, content):
    open(f'./dataset/{file_path}', "w", encoding='utf-8').write(content)
