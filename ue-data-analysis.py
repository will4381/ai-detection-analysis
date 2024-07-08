# !pip install pandas numpy torch transformers textstat nltk spacy scikit-learn tqdm textstat seaborn sns
# !python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import re
from tqdm import tqdm
from textstat import textstat
from collections import Counter
import scipy.stats as stats
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive
from google.colab import userdata

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

drive.mount('/content/drive')

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

def extract_advanced_linguistic_features(text):
    if not isinstance(text, str) or not text.strip():
        return {
            'lexical_diversity': 0,
            'avg_sent_length': 0,
            'parse_tree_depth': 0,
            'flesch_reading_ease': 0,
            'smog_index': 0,
            'discourse_markers': 0
        }

    doc = nlp(text)

    # POS counts
    pos_counts = Counter(token.pos_ for token in doc)

    # Lexical diversity
    lexical_diversity = safe_divide(len(set(token.text.lower() for token in doc)), len(doc))

    # Sentence lengths
    sent_lengths = [len(sent) for sent in doc.sents]
    avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0

    # Syntactic complexity
    parse_tree_depth = max((token.head.i - token.i for token in doc if token.dep_ != "ROOT"), default=0)

    # Named Entity Recognition
    ner_counts = Counter(ent.label_ for ent in doc.ents)

    # Readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    smog_index = textstat.smog_index(text)

    # Discourse markers
    discourse_markers = sum(1 for token in doc if token.dep_ == "discourse")

    return {
        **{f'pos_{k}': v for k, v in pos_counts.items()},
        'lexical_diversity': lexical_diversity,
        'avg_sent_length': avg_sent_length,
        'parse_tree_depth': parse_tree_depth,
        **{f'ner_{k}': v for k, v in ner_counts.items()},
        'flesch_reading_ease': flesch_reading_ease,
        'smog_index': smog_index,
        'discourse_markers': discourse_markers
    }

def extract_paraphrasing_features(original, paraphrased):
    if not isinstance(original, str) or not isinstance(paraphrased, str):
        return {'word_overlap': 0, 'pos_change': 0, 'synonym_usage': 0}

    original_tokens = word_tokenize(original)
    paraphrased_tokens = word_tokenize(paraphrased)

    # Calculate word overlap
    word_overlap = safe_divide(len(set(original_tokens) & set(paraphrased_tokens)),
                               len(set(original_tokens) | set(paraphrased_tokens)))

    # Calculate change in sentence structure
    original_pos = pos_tag(original_tokens)
    paraphrased_pos = pos_tag(paraphrased_tokens)
    pos_change = safe_divide(sum(1 for (_, pos1), (_, pos2) in zip(original_pos, paraphrased_pos) if pos1 != pos2),
                             len(original_pos))

    # Calculate synonym usage
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    synonym_usage = safe_divide(sum(1 for orig, para in zip(original_tokens, paraphrased_tokens)
                                    if para in get_synonyms(orig) and para != orig),
                                len(original_tokens))

    return {'word_overlap': word_overlap, 'pos_change': pos_change, 'synonym_usage': synonym_usage}

# Load your data (add your filepath)
csv_path = '/content/drive/My Drive/ai_text_analysis_results_original.csv'
df = pd.read_csv(csv_path)

data = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    original_text = str(row['original_text'])
    undetectable_text = str(row['undetectable_text'])

    original_features = extract_advanced_linguistic_features(original_text)
    undetectable_features = extract_advanced_linguistic_features(undetectable_text)
    paraphrasing_features = extract_paraphrasing_features(original_text, undetectable_text)

    features = {
        **{f'original_{k}': v for k, v in original_features.items()},
        **{f'undetectable_{k}': v for k, v in undetectable_features.items()},
        **paraphrasing_features
    }

    # Parse the string representations of dictionaries
    try:
        undetectable_scores = ast.literal_eval(row['undetectable_scores'])
        original_scores = ast.literal_eval(row['original_scores'])
        features['score_diff'] = undetectable_scores['ai'] - original_scores['ai']
    except:
        features['score_diff'] = 0

    data.append(features)

df_features = pd.DataFrame(data)

print(f"Processed {len(df_features)} valid rows out of {len(df)} total rows")
print(f"Shape of feature DataFrame: {df_features.shape}")

df_features = df_features.fillna(0)

X = df_features.drop('score_diff', axis=1)
y = df_features['score_diff']

print("Checking for infinite or NaN values:")
print(X.isna().sum().sum())
print(np.isinf(X).sum().sum())
print(np.isnan(y).sum())

print("Checking for infinite or NaN values:")
print(X.isna().sum().sum())
print(np.isinf(X).sum().sum())
print(np.isnan(y).sum())

X_np = X.to_numpy()
y_np = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

class AttentionNet(nn.Module):
    def __init__(self, input_size):
        super(AttentionNet, self).__init__()
        self.attention = nn.Linear(input_size, input_size)
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        weighted_input = x * attention_weights
        x = self.dropout(F.relu(self.bn1(self.fc1(weighted_input))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, attention_weights

class FeatureInteractionNet(nn.Module):
    def __init__(self, input_size):
        super(FeatureInteractionNet, self).__init__()
        self.input_size = input_size
        self.embedding_dim = 16
        self.feature_embedding = nn.Embedding(input_size, self.embedding_dim)
        self.interaction_weight = nn.Parameter(torch.randn(input_size, input_size))
        self.fc = nn.Linear(input_size + input_size * (input_size - 1) // 2, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Compute pairwise interactions
        interactions = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        interactions = interactions.view(batch_size, -1)

        # Use only upper triangular part (excluding diagonal)
        triu_indices = torch.triu_indices(self.input_size, self.input_size, offset=1)
        interactions = interactions[:, triu_indices[0] * self.input_size + triu_indices[1]]

        weighted_interactions = interactions * self.interaction_weight[triu_indices[0], triu_indices[1]]

        combined = torch.cat([x, weighted_interactions], dim=1)

        return self.fc(combined)

class MultiTaskNet(nn.Module):
    def __init__(self, input_size, num_tasks=1):
        super(MultiTaskNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.task_specific_layers = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_tasks)])

    def forward(self, x):
        shared_features = self.shared_layers(x)
        return [task_layer(shared_features) for task_layer in self.task_specific_layers]

attention_model = AttentionNet(X_train_scaled.shape[1]).to(device)
interaction_model = FeatureInteractionNet(X_train_scaled.shape[1]).to(device)
num_tasks = 3
multi_task_model = MultiTaskNet(X_train_scaled.shape[1], num_tasks).to(device)

criterion = nn.MSELoss()
attention_optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=0.001)
multi_task_optimizer = optim.Adam(multi_task_model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

BATCH_SIZE = 32
NUM_WORKERS = 0

def create_data_loader(X, y, batch_size, shuffle):
    tensor_x = torch.FloatTensor(X)
    tensor_y = torch.FloatTensor(y).unsqueeze(1)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=False)

train_loader = create_data_loader(X_train_scaled, y_train, BATCH_SIZE, shuffle=True)
val_loader = create_data_loader(X_val_scaled, y_val, BATCH_SIZE, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    device = next(model.parameters()).device
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10)

    for epoch in tqdm(range(num_epochs), desc="Training", leave=False):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(batch_X)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                    loss = criterion(outputs, batch_y)
                elif isinstance(outputs, list):
                    loss = sum(criterion(output, batch_y) for output in outputs)
                else:
                    loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                with autocast():
                    val_outputs = model(batch_X)
                    if isinstance(val_outputs, tuple):
                        val_outputs, _ = val_outputs
                        val_loss += criterion(val_outputs, batch_y).item()
                    elif isinstance(val_outputs, list):
                        val_loss += sum(criterion(output, batch_y) for output in val_outputs).item()
                    else:
                        val_loss += criterion(val_outputs, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model

attention_model = AttentionNet(X_train_scaled.shape[1]).to(device)
interaction_model = FeatureInteractionNet(X_train_scaled.shape[1]).to(device)
num_tasks = 1  # Set this to the number of tasks you're predicting
multi_task_model = MultiTaskNet(X_train_scaled.shape[1], num_tasks).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

attention_model = attention_model.to(device)
interaction_model = interaction_model.to(device)
multi_task_model = multi_task_model.to(device)

criterion = nn.MSELoss()
attention_optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=0.001)
multi_task_optimizer = optim.Adam(multi_task_model.parameters(), lr=0.001)

try:
    print("Training Attention Model:")
    attention_model = train_model(attention_model, train_loader, val_loader, criterion, attention_optimizer)

    print("\nTraining Interaction Model:")
    interaction_model = train_model(interaction_model, train_loader, val_loader, criterion, interaction_optimizer)

    print("\nTraining Multi-Task Model:")
    multi_task_model = train_model(multi_task_model, train_loader, val_loader, criterion, multi_task_optimizer)
except RuntimeError as e:
    print(f"Error occurred: {str(e)}")
    print("Training failed. Please check your GPU memory usage and model complexity.")

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        if isinstance(model, AttentionNet):
            outputs, _ = model(X)
        elif isinstance(model, MultiTaskNet):
            outputs = model(X)[0]
        else:
            outputs = model(X)
        mse = criterion(outputs, y)
    return mse.item()

attention_mse = evaluate_model(attention_model, X_test_tensor, y_test_tensor)
interaction_mse = evaluate_model(interaction_model, X_test_tensor, y_test_tensor)
multi_task_mse = evaluate_model(multi_task_model, X_test_tensor, y_test_tensor)

print("\nModel Evaluation Results:")
print(f"Attention Model MSE: {attention_mse:.4f}")
print(f"Interaction Model MSE: {interaction_mse:.4f}")
print(f"Multi-Task Model MSE: {multi_task_mse:.4f}")

attention_model.eval()
with torch.no_grad():
    _, attention_weights = attention_model(X_test_tensor)

feature_importance = attention_weights.mean(dim=0).cpu().numpy()

feature_names = list(X.columns)

feature_importance = list(zip(feature_names, feature_importance))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 most important features (Attention Model):")
for feature, importance in feature_importance[:10]:
    print(f"{feature}: {importance:.4f}")

def ablation_study(X, y, feature_groups):
    base_performance = evaluate_model(attention_model, X, y)

    ablation_results = {}

    for group_name, group_indices in feature_groups.items():
        X_ablated = X.clone()
        X_ablated[:, group_indices] = 0

        ablated_performance = evaluate_model(attention_model, X_ablated, y)

        performance_drop = ablated_performance - base_performance
        ablation_results[group_name] = performance_drop

    return ablation_results

# Define feature groups
feature_groups = {
    'POS_tags': [i for i, name in enumerate(feature_names) if any(pos in name for pos in ['NOUN', 'VERB', 'ADJ', 'ADV'])],
    'Readability': [i for i, name in enumerate(feature_names) if any(metric in name for metric in ['readability', 'reading_ease', 'smog_index'])],
    'Syntactic_complexity': [i for i, name in enumerate(feature_names) if any(metric in name for metric in ['parse_tree_depth', 'avg_sent_length'])],
    'Named_entities': [i for i, name in enumerate(feature_names) if 'ner_' in name],
    'Coherence': [i for i, name in enumerate(feature_names) if 'coherence_score' in name],
    'Sentiment': [i for i, name in enumerate(feature_names) if 'sentiment' in name],
    'Paraphrasing': [i for i, name in enumerate(feature_names) if name in ['word_overlap', 'pos_change', 'synonym_usage']]
}

ablation_results = ablation_study(X_test_tensor, y_test_tensor, feature_groups)

print("\nAblation Study Results:")
for group, performance_drop in sorted(ablation_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{group}: Performance drop = {performance_drop:.4f}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_bert_embeddings(text):
    if not isinstance(text, str):
        text = str(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

original_embeddings = np.array([get_bert_embeddings(str(text)) for text in tqdm(df['original_text'], desc="Processing original texts", leave=False)])
undetectable_embeddings = np.array([get_bert_embeddings(str(text)) for text in tqdm(df['undetectable_text'], desc="Processing undetectable texts", leave=False)])

from sklearn.metrics.pairwise import cosine_similarity
similarities = [cosine_similarity(orig.reshape(1, -1), undetectable.reshape(1, -1))[0][0]
                for orig, undetectable in zip(original_embeddings, undetectable_embeddings)]

print("\nBERT-based text similarity analysis:")
print(f"Average cosine similarity: {np.mean(similarities):.4f}")
print(f"Minimum similarity: {np.min(similarities):.4f}")
print(f"Maximum similarity: {np.max(similarities):.4f}")

correlation_matrix = np.corrcoef(np.column_stack((X, similarities)))
feature_names_extended = feature_names + ['BERT_similarity']

print("\nTop correlations with AI detectability score difference:")
correlations = list(zip(feature_names_extended, correlation_matrix[-2]))
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
for feature, correlation in correlations[:10]:
    print(f"{feature}: {correlation:.4f}")

plt.figure(figsize=(12, 8))
sns.barplot(x=[imp for _, imp in feature_importance[:20]], y=[name for name, _ in feature_importance[:20]])
plt.title("Top 20 Features by Importance (Attention Model)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=list(ablation_results.values()), y=list(ablation_results.keys()))
plt.title("Ablation Study Results")
plt.xlabel("Performance Drop")
plt.ylabel("Feature Group")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(similarities, kde=True)
plt.title("Distribution of BERT Cosine Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
combined_embeddings = np.vstack((original_embeddings, undetectable_embeddings))
pca_result = pca.fit_transform(combined_embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(pca_result[:len(original_embeddings), 0], pca_result[:len(original_embeddings), 1], alpha=0.5, label='Original')
plt.scatter(pca_result[len(original_embeddings):, 0], pca_result[len(original_embeddings):, 1], alpha=0.5, label='Undetectable')
plt.title("PCA of BERT Embeddings")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend()
plt.tight_layout()
plt.show()

interaction_model.eval()
with torch.no_grad():
    interaction_outputs = interaction_model(X_test_tensor)
    interaction_weights = interaction_model.fc.weight.cpu().numpy()

n_features = X_test_tensor.shape[1]
interaction_matrix = np.zeros((n_features, n_features))

# Fill the interaction matrix
k = 0
for i in range(n_features):
    for j in range(i+1, n_features):
        interaction_matrix[i, j] = interaction_weights[0, n_features + k]
        interaction_matrix[j, i] = interaction_matrix[i, j]
        k += 1

plt.figure(figsize=(12, 10))
sns.heatmap(interaction_matrix, xticklabels=feature_names, yticklabels=feature_names, cmap='coolwarm', center=0)
plt.title("Feature Interactions")
plt.tight_layout()
plt.show()

multi_task_model.eval()
with torch.no_grad():
    multi_task_outputs = multi_task_model(X_test_tensor)

task_names = ['AI Score', 'Human Score', 'Score Difference']
for i, task_output in enumerate(multi_task_outputs):
    mse = nn.MSELoss()(task_output, y_test_tensor)
    print(f"Task: {task_names[i]}, MSE: {mse.item():.4f}")

shared_features = multi_task_model.shared_layers(X_test_tensor).detach().cpu().numpy()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(shared_features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_test, cmap='viridis')
plt.colorbar(scatter)
plt.title("PCA of Shared Representations in Multi-Task Learning")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.tight_layout()
plt.show()

"""
1. Model Performance:
   - Attention Model MSE: 0.1974
   - Interaction Model MSE: 0.3277
   - Multi-Task Model MSE: 0.2109

   The Attention Model performs the best, followed closely by the Multi-Task Model. The Interaction Model has the highest error, suggesting it might be overfitting or struggling to capture the complexity of the data.

2. Top 10 Important Features (Attention Model):
   The most important features include a mix of named entity recognition (NER) features, part-of-speech (POS) tags, and paraphrasing features:
   - original_ner_FAC and original_ner_PRODUCT: Suggests that the presence of facilities and product names in the original text is a strong indicator.
   - undetectable_pos_ADJ and undetectable_pos_ADV: Indicates that the use of adjectives and adverbs in the undetectable text is important.
   - synonym_usage: Shows that the level of synonym substitution is a key factor in detectability.
   - Presence of interjections (INTJ) and conjunctions (CCONJ) in the original text also play a role.

3. Ablation Study Results:
   - Coherence and Sentiment features don't seem to affect the model's performance significantly.
   - Named entities and Paraphrasing features have the largest negative impact when removed, indicating their importance in detecting AI-generated text.
   - POS tags, Readability, and Syntactic complexity have minor negative impacts when removed.

4. BERT-based Similarity Analysis:
   - The average cosine similarity of 0.8590 suggests that the original and undetectable texts are quite similar semantically.
   - The wide range (0.1176 to 0.9947) indicates varying degrees of success in making texts undetectable.

5. Correlations with AI Detectability Score Difference:
   - Strong correlations (all above 0.99) suggest that many features are highly predictive of the difference in detectability scores.
   - Named entity recognition features (ORG, PERCENT, PRODUCT, WORK_OF_ART) in the undetectable text are highly correlated.
   - Part-of-speech features (PUNCT, AUX, SPACE, SCONJ) also show strong correlations.

Key Insights:
1. Named entity recognition and paraphrasing techniques seem to be crucial in making text less detectable as AI-generated.
2. The use of adjectives and adverbs in the undetectable text plays a significant role.
3. Syntactic features (like punctuation and auxiliary verbs) are important in distinguishing between human and AI-generated text.
4. The high semantic similarity between original and undetectable texts suggests that current methods for making text undetectable focus on preserving meaning while changing surface-level features.
5. The strong correlations across many features indicate that AI detectability is influenced by a wide range of linguistic characteristics, making it a complex problem."""
