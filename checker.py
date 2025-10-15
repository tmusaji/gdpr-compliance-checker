import sys
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import webbrowser

sbert_label_descriptions = {
    0: "Lawfulness, Fairness and Transparency",
    1: "Purpose Limitation",
    2: "Data Minimization",
    3: "Accuracy",
    4: "Storage Limitation",
    5: "Integrity and Confidentiality",
    6: "Accountability",
}

class SBertClassifier(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super(SBertClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_labels)

    def forward(self, embeddings):
        _, (hidden, _) = self.lstm(embeddings.unsqueeze(1))
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(hidden)
        return out

sentence_sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = sentence_sbert_model.get_sentence_embedding_dimension()
device = "cpu"

def load_model(path):
    model = SBertClassifier(embedding_dim, num_labels=7)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def sentence_sbert_classify_policy(policy_sentences, classifier, threshold=0.8):
    results = []
    for sentence in policy_sentences:
        if len(sentence.split()) > 11:
            embedding = sentence_sbert_model.encode(sentence, convert_to_tensor=True).to(device)
            with torch.no_grad():
                outputs = classifier(embedding.unsqueeze(0))
                probs = torch.sigmoid(outputs).squeeze(0)
            sentence_labels = []
            probs = probs.cpu().numpy()
            for idx, score in enumerate(probs):
                label = sbert_label_descriptions.get(idx, "Unknown Label")
                sentence_labels.append((label, score))
            results.append((sentence, sentence_labels))
    return results


def generate_gdpr_report_html(sentence_r):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>GDPR Compliance Report</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f9f9f9; padding: 20px; }
        h1 { text-align: center; color: #333; margin-bottom: 20px; }
        table { width: 95%; margin: 0 auto; border-collapse: collapse; box-shadow: 0 2px 6px rgba(0,0,0,0.1);}
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        tr:hover { background-color: #e6f7ff; }
        td.compliant { color: green; font-weight: bold; }
        td.notaddressed { color: orange; font-weight: bold; }
    </style>
    </head>
    <body>
    <h1>GDPR Compliance Report</h1>
    <table>
    <tr>
        <th>Policy Sentence</th>
        <th>Compliant</th>
        <th>Not Addressed</th>
    </tr>
    """

    for sentence, details in sentence_r.items():
        compliant = ", ".join(details.get("compliant_labels", [])) or "-"
        not_addressed = ", ".join(details.get("not_addressed_labels", [])) or "-"
        html_content += f"<tr><td>{sentence}</td><td class='compliant'>{compliant}</td><td class='notaddressed'>{not_addressed}</td></tr>\n"

    html_content += """
    </table>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python checker.py policy.txt")
        sys.exit(1)

    policy_file = sys.argv[1]
    with open(policy_file, 'r', encoding='utf-8') as file:
        policy_content = file.read()

    sentences = sent_tokenize(policy_content)
    classifier_model = load_model('.models/model.pth')
    classifier_model.to(device)

    policy_results = sentence_sbert_classify_policy(sentences, classifier_model, threshold=0.8)

    sentence_r = {}
    compliance_threshold = 0.8

    for sentence, labels in policy_results:
        compliant_labels = [label for label, score in labels if score >= compliance_threshold]
        not_addressed_labels = [label for label, score in labels if score < compliance_threshold]

        sentence_r[sentence] = {
            "compliant_labels": compliant_labels,
            "not_addressed_labels": not_addressed_labels
        }

    html_content = generate_gdpr_report_html(sentence_r)
    html_file_path = f'{policy_file.rsplit(".", 1)[0]}_Compliance_Report.html'
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    webbrowser.open(f'file://{html_file_path}')
