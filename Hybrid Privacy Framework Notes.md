Hybrid Privacy Framework.py is a Python script for privacy-preserving data processing. Its main purpose is to detect and protect personally identifiable information (PII) in datasets using pseudonymization and anonymization techniques.

What it does:

Loads a dataset (e.g., retail or ad-click data).
Uses a SpaCy Named Entity Recognition (NER) model to detect PII such as names, emails, phone numbers, and addresses.
Replaces PII with pseudonyms (tokens or salted hashes).
Applies anonymization methods like k-anonymity (e.g., grouping ages, generalizing ZIP codes).
Saves the processed, privacy-protected data for further analysis.
How to use it:

Make sure your dataset (CSV) is available and adjust the DATA_PATH variable to point to your file.
Install the dependencies:
pip install spacy pandas scikit-learn faker hashlib tqdm
If you have a custom-trained SpaCy NER model, update the SPACY_MODEL variable to load it.
Run the script. It will:
Load your data,
Split it for training/testing,
Detect and pseudonymize PII,
Apply anonymization,
Save the processed output to the path specified in OUTPUT_PATH.
The output will be a new CSV file with sensitive information protected.
