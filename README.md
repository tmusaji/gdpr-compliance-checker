# GDPR Compliance Checker

A Python-based tool used for analysis of the policies and privacy documents to check their compliance with the GDPR principles.

It uses Sentence-BERT (SBERT) embeddings and a pretrained PyTorch model (.pth) to classify policies to their respective principles and where a sentence is not addressing the principle.

# Features

==> Tokenizes each policy sentence into embeddings (tensors). 

==> Classify based on the seven GDPR principles.

==> Identifies principles that the policy sentence does not address.

==> Generates HTML compliance report.

==> Uses a pretrained model for classification.

# Installation

git clone https://github.com/tmusaji/gdpr-compliance-checker.git

cd gdpr-compliance-checker

pip install -r requirements.txt

# Usage

python checker.py sample_policy.txt

# Example 

Input: "We collect personal data only with the user’s consent and always explain why and how it will be used."

Compliant: "Lawfulness, Fairness and Transparency, Purpose Limitation, Data Minimization"

Not Addressed: "Accuracy, Storage Limitation, Integrity and Confidentiality, Accountability"

# Folder Structure

gdpr-compliance-checker/
│
├── checker.py             
├── requirements.txt       
├── README.md             
├── models/                
          └── model.pth
          
├── examples/       
             └── sample_policy.txt
             
└── reports/
             └── sample_Compliance_Report.html     

# Screenshots 

<img width="1272" height="458" alt="Screenshot 2025-10-15 105251" src="https://github.com/user-attachments/assets/7f14ca45-faa8-4953-b065-4c9f9f3ff2ed" />
