import pandas as pd

proteomics_df = pd.read_csv(r"C:\My_Data\EMBL\Skin\20231023_092657_imputed_proteomics.csv")
transcriptomics_df  = pd.read_csv(r"C:\My_Data\EMBL\Skin\20231023_092657_imputed_transcriptomics.csv")

protein_names = set(proteomics_df.columns)
common_genes = [gene for gene in transcriptomics_df.columns if gene in protein_names]

filtered_transcriptomics_df = transcriptomics_df[common_genes]

common_columns = list(set(proteomics_df.columns).intersection(set(transcriptomics_df.columns)))

# Filter both DataFrames to only these common features
filtered_proteomics_df1 = proteomics_df[common_columns]
filtered_transcriptomics_df1 = transcriptomics_df[common_columns]

filtered_proteomics_df = filtered_proteomics_df1[['Unnamed: 0'] + [col for col in filtered_proteomics_df1.columns if col != 'Unnamed: 0']]
filtered_transcriptomics_df = filtered_transcriptomics_df1[['Unnamed: 0'] + [col for col in filtered_transcriptomics_df1.columns if col != 'Unnamed: 0']]



train_df_proteomics = filtered_proteomics_df.iloc[:40]
val_df_proteomics = filtered_proteomics_df.iloc[40:]

train_df_transcriptomics = filtered_transcriptomics_df.iloc[:40]
val_df_transcriptomics = filtered_transcriptomics_df.iloc[40:]


import torch
from torch.utils.data import Dataset, DataLoader

class MultiOmicsDataset(Dataset):
    def __init__(self, proteomics_df, transcriptomics_df):
        # Store subject IDs
        self.subject_ids = proteomics_df["Unnamed: 0"].values
        
        # Store feature data as tensors (excluding the ID column)
        self.proteomics_data = torch.tensor(proteomics_df.drop(columns=['Unnamed: 0']).values, dtype=torch.float32)
        self.transcriptomics_data = torch.tensor(transcriptomics_df.drop(columns=['Unnamed: 0']).values, dtype=torch.float32)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        return self.subject_ids[idx], self.proteomics_data[idx], self.transcriptomics_data[idx]
        
    
# Create dataset objects
train_dataset = MultiOmicsDataset(train_df_proteomics, train_df_transcriptomics)
val_dataset = MultiOmicsDataset(val_df_proteomics, val_df_transcriptomics)

# Create dataloaders with batch size 4
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


# a1 = iter(train_loader)
# a2 = next(a1)
# name = a2[0]
# p =a2[1].numpy()
# t = a2[2].numpy()

import torch
import torch.nn as nn

class TranscriptToProteomicsTransformer(nn.Module):
    def __init__(self, feature_dim=4886, d_model=256, nhead=2, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        # 1. Linear projection: embed each feature (token) into d_model
        self.input_proj = nn.Linear(1, d_model)

        # 2. Positional encoding: optional but helps capture position
        self.pos_embedding = nn.Parameter(torch.randn(1, feature_dim, d_model))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Project back to 1 value per feature
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [B, feature_dim] → we treat each feature as a token
        Output: [B, feature_dim]
        """
        B, F = x.shape

        # Treat each feature as a token: reshape to [B, F, 1]
        x = x.unsqueeze(-1)

        # Project to transformer space: [B, F, d_model]
        x = self.input_proj(x)

        # Add position encoding
        x = x + self.pos_embedding[:, :F, :]

        # Transformer encoder
        x = self.transformer_encoder(x)  # [B, F, d_model]
        
        # Back to scalar predictions per feature
        x = self.output_proj(x).squeeze(-1)  # [B, F]
        return x

def model() -> TranscriptToProteomicsTransformer:
    model = TranscriptToProteomicsTransformer()
    return model

# batch_size = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model()

x = torch.randn(4, 48).to(DEVICE)
out = model(x)
#print(out.shape)  # Should be: torch.Size([4, 4886])











