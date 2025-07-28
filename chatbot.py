# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator

app = Flask(__name__)

model_name = "Helsinki-NLP/opus-mt-en-mr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

try:
    chatbot_model = pipeline("text-generation", model="gpt2")
except Exception as e:
    chatbot_model = None  # Model loading failed

translator = Translator()

def generate_marathi_response(prompt):
    if chatbot_model is None:
        return "क्षमा करा, चॅटबॉट सध्या उपलब्ध नाही."  # Chatbot not available

    try:
        translated_input = translator.translate(prompt, src="mr", dest="en").text

        # Truncate the translated input to 512 tokens
        tokenized_input = tokenizer.encode(translated_input, return_tensors='pt')
        truncated_input = tokenizer.decode(tokenized_input[0][:512])

        english_response = chatbot_model(truncated_input, max_length=150, do_sample=True, truncation=True)[0]['generated_text']  # Added truncation=True

        inputs = tokenizer(english_response, return_tensors="pt")
        outputs = model.generate(**inputs)
        marathi_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return marathi_response
    except IndexError:
        return "मला प्रतिसाद तयार करता आला नाही."  # Or a more informative error message
    except Exception as e:
        return f"त्रुटी: {str(e)}"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_data = request.get_json()
        marathi_input = user_data.get("message")
        if not marathi_input:
            return jsonify({"error": "कृपया संदेश द्या."})  # Handle empty input

        marathi_response = generate_marathi_response(marathi_input)
        return jsonify({"response": marathi_response})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5000)