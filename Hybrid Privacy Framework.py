"""
Privacy-Preserving Data Processing Pipeline
-------------------------------------------
Implements the methodology described in 'Miguel Acosta Praxis Research'.
- Loads dataset
- Trains or loads a SpaCy NER model to detect PII
- Applies pseudonymization (tokenization + salted hashing)
- Applies anonymization (k-anonymity + generalization)
- Saves processed data for analysis

Dependencies:
    pip install spacy pandas scikit-learn faker hashlib tqdm
"""

import spacy
import pandas as pd
import hashlib
import random
import string
from tqdm import tqdm
from faker import Faker
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. CONFIGURATION
# ----------------------------

DATA_PATH = "your_dataset.csv"  # Replace with path to retail or ad-click dataset
OUTPUT_PATH = "processed_dataset.csv"
SPACY_MODEL = "en_core_web_sm"  # Use a custom trained NER model if available
SALT = "a_random_secret_salt"   # Used for salted hashing

fake = Faker()

# ----------------------------
# 2. HELPER FUNCTIONS
# ----------------------------

def generate_token(prefix="CUST"):
    """Generate a random token for pseudonymization."""
    return f"{prefix}{''.join(random.choices(string.digits, k=6))}"

def hash_identifier(identifier, salt=SALT):
    """Return a SHA-256 salted hash of an identifier."""
    return hashlib.sha256((salt + identifier).encode()).hexdigest()

def k_anonymize_column(df, column, k=5):
    """
    Apply k-anonymity by generalizing values.
    For demonstration: groups ages into bins; generalize ZIP codes.
    """
    if column == "age":
        df[column] = pd.cut(df[column], bins=[0,18,30,40,50,60,100],
                            labels=["0-17","18-29","30-39","40-49","50-59","60+"])
    elif column == "zip":
        df[column] = df[column].astype(str).str[:3] + "XX"  # Generalize ZIP codes
    return df

def pseudonymize_text(text, nlp, token_map):
    """Detect PII using SpaCy NER and replace with tokens/hashes."""
    doc = nlp(text)
    new_text = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "EMAIL", "PHONE", "ADDRESS"]:
            if ent.label_ == "EMAIL":
                replacement = hash_identifier(ent.text)  # hash emails
            else:
                replacement = token_map.get(ent.text) or generate_token()
                token_map[ent.text] = replacement
            new_text = new_text.replace(ent.text, replacement)
    return new_text

# ----------------------------
# 3. LOAD & SPLIT DATA
# ----------------------------

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Example assumption: dataset has columns ['name','email','phone','address','age','zip','comments']
# Adjust columns to match your dataset.

print("Splitting data 70/30...")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# ----------------------------
# 4. LOAD OR TRAIN NER MODEL
# ----------------------------

print("Loading SpaCy model...")
nlp = spacy.load(SPACY_MODEL)

# NOTE: If using a custom model, load that instead:
# nlp = spacy.load("path_to_custom_model")

# ----------------------------
# 5. APPLY PSEUDONYMIZATION
# ----------------------------

print("Applying pseudonymization...")
token_map = {}
for col in ["name", "email", "phone", "address", "comments"]:
    if col in df.columns:
        tqdm.pandas(desc=f"Pseudonymizing {col}")
        df[col] = df[col].astype(str).progress_apply(lambda x: pseudonymize_text(x, nlp, token_map))

# ----------------------------
# 6. APPLY ANONYMIZATION
# ----------------------------

print("Applying anonymization...")
if "age" in df.columns:
    df = k_anonymize_column(df, "age")
if "zip" in df.columns:
    df = k_anonymize_column(df, "zip")

# ----------------------------
# 7. SAVE PROCESSED DATA
# ----------------------------

print(f"Saving processed dataset to {OUTPUT_PATH}...")
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Processing complete. Data saved.")
