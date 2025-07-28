# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
import logging

app = Flask(__name__)

model_name = "Helsinki-NLP/opus-mt-en-mr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

try:
    chatbot_model = pipeline("text-generation", model="gpt2") # Consider fine-tuning GPT-2
except Exception as e:
    logging.error(f"Error loading GPT-2: {e}")
    chatbot_model = None

translator = Translator()

def generate_marathi_response(prompt):
    if chatbot_model is None:
        return "क्षमा करा, चॅटबॉट सध्या उपलब्ध नाही."

    try:
        translated_input = translator.translate(prompt, src="mr", dest="en").text
        english_response = chatbot_model(translated_input, max_length=150, do_sample=True)[0]['generated_text']

        inputs = tokenizer(english_response, return_tensors="pt")
        outputs = model.generate(**inputs)
        marathi_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return marathi_response
    except IndexError:
        logging.error("IndexError during response generation.")
        return "मला प्रतिसाद तयार करता आला नाही."
    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        return "एक तांत्रिक समस्या आली आहे."

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_data = request.get_json()
        marathi_input = user_data.get("message")
        if not marathi_input:
            return jsonify({"error": "कृपया संदेश द्या."})

        marathi_response = generate_marathi_response(marathi_input)
        return jsonify({"response": marathi_response})
    except Exception as e:
        logging.error(f"Error in chatbot endpoint: {e}")
        return jsonify({"error": "एक तांत्रिक समस्या आली आहे."})

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    app.run(port=5001)