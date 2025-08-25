Banking Complaint Severity Predictor (LoRA Fine-Tuned DistilBERT)

________________________________________
Overview
The Banking Complaint Severity Predictor is an AI-powered system that automatically classifies customer banking complaints into severity levels. Instead of just mapping complaints to broad categories (like loan, credit card, etc.), this model predicts priority levels (Low → Extreme Severe) from complaint text.
This helps financial institutions triage issues faster, giving urgent complaints (fraud, large unauthorized withdrawals) immediate attention, while handling routine ones with normal workflows.
The system was built by fine-tuning DistilBERT with LoRA adapters on the Consumer Financial Protection Bureau (CFPB) complaint dataset, and deployed as a Flask web app with an interactive frontend.
________________________________________
Dataset Used
-Consumer Financial Protection Bureau (CFPB) Consumer Complaints Dataset
-Approx 22000 Rows of data
Features
Complaint Ingestion & Preprocessing
•	Reads CFPB complaints dataset (CSV).
•	Cleans and filters only complaints with customer narratives.
•	Handles severe class imbalance via oversampling for minority classes.
Severity Labeling
•	Maps each complaint into six severity levels:
o	Extreme Low
o	Low
o	Medium
o	Extreme Medium
o	Severe
o	Extreme Severe
•	Threshold-based logic improves interpretation of model probabilities, ensuring nuanced severity classification.
Transformer Fine-Tuning with LoRA
•	Uses DistilBERT-base-uncased as backbone.
•	Fine-tuned with LoRA adapters (PEFT) to reduce GPU memory usage and training cost.
•	Implemented early stopping and class weights to handle imbalance.
•	Achieved ~76% accuracy and improved macro-F1 after oversampling (vs ~66% before).
Flask Web Application
•	Input: Free-text complaint typed/pasted by user.
•	Output: Predicted severity + confidence score.
•	Stylish UI with color-coded severity categories.
________________________________________
System Architecture
User (Bank Staff / Complaint Officer) →
Flask Web App (Frontend: HTML/CSS, Bootstrap) →
Preprocessing & Tokenization (HuggingFace AutoTokenizer) →
Fine-Tuned DistilBERT + LoRA (PyTorch, PEFT) →
Threshold Mapping → Severity Level + Score →
Flask → HTML Frontend
________________________________________
Tech Stack
Backend: Flask, Python
ML / NLP: HuggingFace Transformers, PyTorch, PEFT (LoRA), Scikit-learn
Frontend: HTML, CSS, Bootstrap (styled complaint submission UI)
Data: CFPB Consumer Complaints Dataset (~1.6 GB raw)
Deployment: Flask server (local), extendable to cloud (AWS, GCP, Azure)
________________________________________
Project Structure
banking-complaint-severity/
│── app. ipynb              # Data exploration, training pipeline
│── backend.py             # Flask backend
│── templates/
│    └── index.html        # Frontend (UI for classifier)
│── complaints.csv         # Original CFPB dataset (1.6 GB)
│── results/               # Training checkpoints
│── banking_severity_model_merged/ # Final merged model for inference
│── requirements.txt       # Dependencies
________________________________________
Challenges & Learnings
Dataset Size & NaNs
•	Problem: Original dataset (~1.6 GB) too large for Excel / direct loading, with many missing complaint narratives.
•	Solution: Loaded incrementally in pandas (nrows), dropped non-narrative complaints (~80% NaNs), resulting in ~22k usable rows.
Class Imbalance
•	Problem: High-severity cases were rare (<3%). Model biased towards Medium.
•	Solution: Applied oversampling + class weights during training. Accuracy improved from ~66% → 76%, macro-F1 improved significantly.
Checkpoint Mismatches (LoRA vs HuggingFace)
•	Problem: Loading LoRA checkpoints caused size mismatch errors (3 vs 2 labels).
•	Solution: Forced num_labels=3 during base model init, then merged LoRA adapters with merge_and_unload () to save a clean HuggingFace model for inference.
Early Stopping Behavior
•	Problem: Model sometimes stopped too early on minor validation fluctuations.
•	Solution: Tuned patience and thresholds; learned that early stopping counts relative to the best loss seen, not just previous epoch.
Deployment Issues
•	Problem: Backend initially loaded PEFT checkpoints incorrectly.
•	Solution: Merged LoRA weights → exported plain HuggingFace model → Flask loads directly without PEFT.
Classification Refinement
•	Problem: Severe complaints often misclassified as Medium due to probability overlap.
•	Solution: Introduced threshold bands + Extreme categories to better map probabilities into nuanced severities.
________________________________________
Author
Akash Gupta
M.Tech Artificial Intelligence, IIT Kharagpur

