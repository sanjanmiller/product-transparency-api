from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ✅ Load model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

app = Flask(__name__)

# ✅ Dynamic Question Generation
def generate_dynamic_question(product_name, previous_answers=[]):
    prev_text = ", ".join(previous_answers) if previous_answers else "None"
    instruction = (
        "Generate ONE intelligent follow-up question for a Product Transparency Form. "
        "Focus on origin, sourcing, ethical production, certifications, sustainability, or ingredients."
    )
    examples = """
    Product: Organic Mango. Previous answers: None
    Follow-up Question: Where were these mangoes grown, and are they certified organic?

    Product: Fresh Spinach. Previous answers: Grown without pesticides
    Follow-up Question: What sustainable farming practices were used for this spinach, and is it certified pesticide-free?

    Product: Grass-Fed Cow Milk. Previous answers: Cows are grass-fed year-round
    Follow-up Question: Is this milk certified grass-fed, and are the cows given any supplementary feed?
    """
    input_text = f"""{instruction}

    {examples}

    Product: {product_name}. Previous answers: {prev_text}
    Follow-up Question:"""

    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=80, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Transparency Score Logic
origin_keywords = ["grown", "harvested", "produced", "origin", "farm", "region", "country"]
certification_keywords = ["certified", "organic", "fair trade", "verified"]
sustainability_keywords = ["pesticide-free", "sustainably", "eco-friendly", "ethical", "grass-fed"]
ingredient_keywords = ["ingredients", "composition", "contains", "made of"]

def calculate_transparency_score(product_name, previous_answers):
    text = " ".join(previous_answers).lower()
    score = 0
    if any(word in text for word in origin_keywords): score += 25
    if any(word in text for word in certification_keywords): score += 25
    if any(word in text for word in sustainability_keywords): score += 25
    if any(word in text for word in ingredient_keywords): score += 25

    if score >= 75:
        msg = "High transparency. Great detail provided."
    elif score >= 50:
        msg = "Good transparency. More details can be added."
    elif score > 0:
        msg = "Low transparency. Needs more origin and certification details."
    else:
        msg = "No transparency data provided."

    return {"product_name": product_name, "transparency_score": score, "message": msg}

# ✅ Home Route
@app.route("/")
def home():
    return "<h2>✅ Product Transparency API is Running</h2><p>Use /generate-questions or /transparency-score</p>"

# ✅ Generate Questions Endpoint
@app.route("/generate-questions", methods=["GET", "POST"])
def generate_questions():
    if request.method == "POST":
        data = request.json
        product_name = data.get("product_name", "")
        previous_answers = data.get("previous_answers", [])
    else:  # GET method
        product_name = request.args.get("product_name", "")
        previous_answers = request.args.getlist("answers")

    question = generate_dynamic_question(product_name, previous_answers)
    return jsonify({"product_name": product_name, "follow_up_question": question})

# ✅ Transparency Score Endpoint
@app.route("/transparency-score", methods=["GET", "POST"])
def transparency_score():
    if request.method == "POST":
        data = request.json
        product_name = data.get("product_name", "")
        previous_answers = data.get("previous_answers", [])
    else:  # GET method
        product_name = request.args.get("product_name", "")
        previous_answers = request.args.getlist("answers")

    return jsonify(calculate_transparency_score(product_name, previous_answers))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
