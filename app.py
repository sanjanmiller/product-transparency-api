import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Question generation logic
def generate_dynamic_question(product_name, previous_answers=""):
    previous_answers = [a.strip() for a in previous_answers.split(",") if a.strip()]
    prev_text = ", ".join(previous_answers) if previous_answers else "None"

    instruction = (
        "You are an assistant that generates ONE intelligent follow-up question for a Product Transparency Form.\n\n"
        "Rules:\n"
        "1. Only ONE question — no lists or multiple-choice.\n"
        "2. Focus ONLY on: origin, sourcing, ethical production, certifications, sustainability, or ingredients.\n"
        "3. Avoid repeating known information already in the previous answers.\n"
        "4. No generic or quiz-style questions (e.g., 'Which of the following…', 'Is this a product?').\n"
        "5. Use professional tone and proper grammar.\n\n"
        "---"
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


# Transparency score logic
origin_keywords = ["grown", "harvested", "produced", "origin", "farm", "region", "country"]
certification_keywords = ["certified", "organic", "fair trade", "verified"]
sustainability_keywords = ["pesticide-free", "sustainably", "eco-friendly", "ethical", "grass-fed"]
ingredient_keywords = ["ingredients", "composition", "contains", "made of"]

def calculate_transparency_score(product_name, previous_answers=""):
    previous_answers = [a.strip() for a in previous_answers.split(",") if a.strip()]
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

    return f"Transparency Score: {score}/100\n{msg}"

# Gradio UI
question_interface = gr.Interface(
    fn=generate_dynamic_question,
    inputs=[
        gr.Textbox(label="Product Name"),
        gr.Textbox(label="Previous Answers (comma-separated)"),
    ],
    outputs="text",
    title="🧠 Intelligent Follow-up Question Generator",
    description="Generate a smart question based on product info and previous answers."
)

score_interface = gr.Interface(
    fn=calculate_transparency_score,
    inputs=[
        gr.Textbox(label="Product Name"),
        gr.Textbox(label="Previous Answers (comma-separated)"),
    ],
    outputs="text",
    title="📊 Transparency Score Calculator",
    description="Calculate how transparent a product's info is."
)

demo = gr.TabbedInterface(
    interface_list=[question_interface, score_interface],
    tab_names=["Generate Follow-up Question", "Transparency Score"]
)

demo.launch()
