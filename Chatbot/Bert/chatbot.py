import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler
import torch.nn.functional as F 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes

# specify CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data from Excel
df = pd.read_excel("chitchat.xlsx", sheet_name="questions")
originalDf = df

# Convert all labels to string
df['label'] = df['label'].astype(str)

# Case folding (lowercasing)
df['text'] = df['text'].str.lower()

# Converting the labels into encodings
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Check class distribution
print(df['label'].value_counts(normalize=True))

train_text, train_labels = df['text'], df['label']

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

max_seq_len = 18

# Tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# Convert tokens to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# Define a batch size
batch_size = 18

# Wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# Sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# DataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(np.unique(train_labels)))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False

model = BERT_Arch(bert)
model = model.to(device)

from torchinfo import summary
summary(model)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-3)

# Compute the class weights
class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
print(class_wts)

# Training and validation loss lists
train_losses = []

# Number of training epochs
epochs = 200

# Learning rate scheduler
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

def train():
    model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)
        loss = F.nll_loss(preds, labels)
        total_loss = total_loss + loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss, _ = train()
    train_losses.append(train_loss)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f'\nTraining Loss: {train_loss:.3f}')

def get_prediction(input_text):
    input_text = re.sub(r'[^a-zA-Z ]+', '', input_text)
    test_text = [input_text]
    model.eval()

    tokens_test_data = tokenizer(
        test_text,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

    probs = np.exp(preds)
    probs = probs / np.sum(probs)
    max_prob = np.max(probs)
    pred_class = np.argmax(probs, axis=1)

    print(f"Text: {input_text}, Probabilities: {probs}")

    return le.inverse_transform(pred_class)[0], max_prob

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Halo! Saya adalah bot Thoriqul Jannah. Bagaimana saya bisa membantu Anda?")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    classification, confidence = get_prediction(text)
    print(f'Classification: {classification}, Confidence: {confidence}')

    confidence_threshold = 0.99  # Adjust this threshold as needed

    # Load responses from Excel
    response_df = pd.read_excel("chitchat.xlsx", sheet_name="responses")
    response_dict = dict(zip(response_df['label'], response_df['response']))

    if confidence > confidence_threshold:
        response = response_dict.get(classification, "Maaf, saya tidak mengerti pertanyaan Anda.")
    else:
        response = "Maaf, saya tidak mengerti pertanyaan Anda."

    await update.message.reply_text(response)

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

testDf = pd.read_excel("chitchat-test.xlsx", sheet_name="questions")

test_labels = testDf['label']
test_data = testDf['text']

def evaluate_model():
    model.eval()

    # Preprocessing the test texts
    tokens_test_data = tokenizer(
        test_data.tolist(),
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

    # Calculate probabilities and predicted classes
    probs = np.exp(preds)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    pred_classes = np.argmax(probs, axis=1)

    # Convert predictions to original labels
    pred_labels = le.inverse_transform(pred_classes)
    # Calculate accuracy
    accuracy = accuracy_score(test_labels.tolist(), pred_labels)

    report = classification_report(test_labels.tolist(), pred_labels)

    print(f"Accuracy: {accuracy * 100}%")
    print(f"Classification Report:\n{report}")

    # Confusion Matrik

    cm = confusion_matrix(test_labels.tolist(), pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')


    return accuracy, report

accuracy, report = evaluate_model()

def main():
    application = Application.builder().token("7495537275:AAE--GCDpRX7_YPzs8GZqTBrhAueqAd1BC0").build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    application.add_handler(MessageHandler(None, echo))

    application.run_polling()

if __name__ == '__main__':
    main()
