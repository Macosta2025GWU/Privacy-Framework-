"""
Validation Module for Privacy-Preserving Pipeline
-------------------------------------------------
Evaluates:
1. NER Performance (Precision, Recall, F1)
2. Privacy Metrics (Re-identification risk proxy)
3. Data Utility Metrics (Post-processing utility)
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------------
# 1. NER VALIDATION METRICS
# ----------------------------

def validate_ner(test_texts, true_labels, nlp):
    """
    Evaluate NER model on held-out texts.
    test_texts: list of strings
    true_labels: list of dicts with true entity spans (start, end, label)
    """
    y_true, y_pred = [], []
    for text, labels in zip(test_texts, true_labels):
        doc = nlp(text)
        predicted_ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        
        # Convert to binary (PII or not)
        for start, end, label in labels:
            y_true.append(1)  # 1 = PII present
            if any((p[0] == start and p[1] == end) for p in predicted_ents):
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        for p in predicted_ents:
            if not any((l[0] == p[0] and l[1] == p[1]) for l in labels):
                # False positive
                y_true.append(0)
                y_pred.append(1)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"NER Validation → Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    return precision, recall, f1

# ----------------------------
# 2. PRIVACY METRICS
# ----------------------------

def reidentification_risk(df, pii_columns):
    """
    Simple proxy: check how many records remain uniquely identifiable.
    Lower % unique = lower risk.
    """
    total_records = len(df)
    risky_records = 0
    for col in pii_columns:
        unique_vals = df[col].nunique()
        if unique_vals > total_records * 0.9:  # Too many unique identifiers
            risky_records += 1
    risk_score = (risky_records / len(pii_columns)) * 100
    print(f"Privacy Metric → Re-identification Risk Proxy: {risk_score:.1f}%")
    return risk_score

# ----------------------------
# 3. DATA UTILITY METRICS
# ----------------------------

def utility_check(original_df, processed_df, non_pii_columns):
    """
    Measures how much data utility remains by comparing non-PII columns
    (categorical accuracy preservation).
    """
    preserved = 0
    for col in non_pii_columns:
        preserved += (original_df[col] == processed_df[col]).mean()
    avg_preservation = preserved / len(non_pii_columns)
    print(f"Data Utility Metric → Average Preservation: {avg_preservation*100:.1f}%")
    return avg_preservation

# ----------------------------
# 4. SAMPLE USAGE
# ----------------------------

if __name__ == "__main__":
    # Assume we already have train/test splits & processed data from main script
    raw_data = pd.read_csv("your_dataset.csv")
    processed_data = pd.read_csv("processed_dataset.csv")

    # Example: Evaluate privacy
    reidentification_risk(processed_data, pii_columns=["name", "email", "phone"])

    # Example: Evaluate utility (non-PII fields)
    utility_check(raw_data, processed_data, non_pii_columns=["transaction_history", "clicked"])
