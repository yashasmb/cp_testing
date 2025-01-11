import torch
import streamlit as st
from torch import nn
import numpy as np

# Define the model architecture again (same as before)
class CpGPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CpGPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(6, input_size)  # 6 because we have 5 possible values (A, C, G, T, N, 0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load the model (make sure the model path is correct)
model = CpGPredictor(input_size=128, hidden_size=256, num_layers=2, output_size=1)
model.load_state_dict(torch.load("model_var_len.pth"))
model.eval()

# Streamlit app
st.title("CpG Count Prediction")

# Define nucleotide to integer mapping
nucleotide_map = {
    'A': 1,
    'C': 2,
    'G': 3,
    'T': 4,
    'N': 0
}

# Input: DNA sequence
sequence_input = st.text_area("Enter DNA Sequence (e.g., 'ACGTN')")

if sequence_input:
    # Convert the sequence to a list of integers based on the mapping
    sequence = [nucleotide_map.get(base, 0) for base in sequence_input.strip().upper()]

    # Pad the sequence if necessary
    max_len = 128  # max length of sequences in training
    padded_sequence = sequence + [0] * (max_len - len(sequence))

    # Convert to tensor
    input_tensor = torch.tensor(padded_sequence).unsqueeze(0)

    # Predict CpG count using the model
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_cpg_count = prediction.item()

    # Display the result
    st.write(f"Predicted CpG Count: {predicted_cpg_count:.2f}")
