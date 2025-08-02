# 🟢 Product Transparency API  

This project is an **AI-powered Flask microservice** that generates dynamic follow-up questions for product transparency and calculates a transparency score based on user-provided answers.  

## 📌 Features  
✔️ **/generate-questions** → Generates intelligent product-specific follow-up questions using T5.  
✔️ **/transparency-score** → Returns a transparency score based on provided answers.  
✔️ Built with **Flask** and **Transformers (T5)**.  

---

## 📷 Screenshots  

### 🏠 Home Endpoint  
<img src="https://raw.githubusercontent.com/sanjanmiller/product-transparency-api/main/output.png" width="600">

### ❓ Generate Questions  
<img src="https://raw.githubusercontent.com/sanjanmiller/product-transparency-api/main/Screenshot%20(390).png" width="600">

### 📊 Transparency Score  
<img src="https://raw.githubusercontent.com/sanjanmiller/product-transparency-api/main/Screenshot%20(391).png" width="600">

---

## 📌 How to Run

```bash
# 1️⃣ Clone the repo
git clone https://github.com/sanjanmiller/product-transparency-api.git
cd product-transparency-api

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the app
python app.py

# 4️⃣ Access the API in your browser or Postman:
# ✅ Home
http://127.0.0.1:5000/

# ✅ Generate Questions (GET)
http://127.0.0.1:5000/generate-questions?product_name=Organic%20Mango&answers=Harvested%20in%20Kerala

# ✅ Transparency Score (GET)
http://127.0.0.1:5000/transparency-score?product_name=Organic%20Mango&answers=Harvested%20in%20Kerala&answers=Certified%20Organic

