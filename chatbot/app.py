from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize the Flask app
app = Flask(__name__)

# Route to serve the chat interface
@app.route("/")
def index():
    return render_template('chat.html')

# Route to handle chat input and generate responses
@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get('msg')  # Get the message from the AJAX request
    
    if user_input:
        response = get_chat_response(user_input)  # Get the chatbot's response
        return jsonify({'response': response})   # Return it as JSON
    else:
        return jsonify({'response': "I didn't understand that."})

# Function to generate chatbot response
def get_chat_response(text):
    global chat_history_ids  # Declare it as global

    # Tokenize the user input and append to chat history if it's not the first message
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # Check if chat_history_ids exists, if not, initialize it
    if 'chat_history_ids' not in globals():
        chat_history_ids = new_user_input_ids
    else:
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate the chatbot response, limiting the chat history to 1000 tokens
    chat_history_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the bot's response
    return tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)


# Run the Flask app
if __name__ == '__main__':
    app.run()
