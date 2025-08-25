from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Load merged model (clean HuggingFace version, no PEFT)
model_path = "./banking_severity_model_merged"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    severity = None
    score = None
    text = ""
    if request.method == "POST":
        text = request.form["complaint"]
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        # Run prediction
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Extract class probabilities
        low_prob, med_prob, high_prob = probs[0], probs[1], probs[2]
        top_class = probs.argmax()
        # Map to custom severity levels with thresholds
        if top_class == 2:  # High
            severity = "Extreme Severe" if high_prob > 0.75 else "Severe"
            score = high_prob

        elif top_class == 1:  # Medium
            severity = "Extreme Medium" if med_prob > 0.75 else "Medium"
            score = med_prob

        else:  # Low
            severity = "Extreme Low" if low_prob > 0.75 else "Low"
            score = low_prob

        score = round(score*100, 2)

        # Convert to percentage for display
        

    return render_template("index.html", severity=severity, score=score, text=text)

if __name__ == "__main__":
    app.run(debug=True)

