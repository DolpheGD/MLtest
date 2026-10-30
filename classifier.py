"""
Install:  pip install torch pandas scikit-learn
Run:      python classifier.py
"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 1. DATA
URL = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
df = pd.read_csv(URL)

FEATURES = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
X = df[FEATURES].values.astype("float32")
y = (df["Legendary"].astype(str).str.upper() == "TRUE").values.astype("float32")

print(f"{len(df)} Pokémon, {int(y.sum())} Legendary ({y.mean():.1%})")

# Split, then scale (fit the scaler on training data only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# Convert to tensors and wrap in DataLoaders
def to_loader(X, y, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    return DataLoader(ds, batch_size=32, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
test_loader = to_loader(X_test, y_test, shuffle=False)

# --------------------------------------------------------------- 2. MODEL
model = nn.Sequential(
    nn.Linear(6, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),   # one raw score (a "logit")
)

# Legendaries are rare (~8%), so up-weight them in the loss.
pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()], dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------------------------------------- 3. TRAIN
for epoch in range(1, 41):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()           # clear old gradients
        loss = loss_fn(model(xb), yb)   # forward pass + loss
        loss.backward()                 # backpropagation
        optimizer.step()                # update weights
        total += loss.item() * len(xb)
    if epoch % 10 == 0:
        print(f"epoch {epoch:2d}  loss {total / len(train_loader.dataset):.4f}")

# ------------------------------------------------------------ 4. EVALUATE
model.eval()
tp = fp = fn = correct = n = 0
with torch.no_grad():
    for xb, yb in test_loader:
        pred = (torch.sigmoid(model(xb)) > 0.5).float()
        correct += (pred == yb).sum().item()
        tp += ((pred == 1) & (yb == 1)).sum().item()
        fp += ((pred == 1) & (yb == 0)).sum().item()
        fn += ((pred == 0) & (yb == 1)).sum().item()
        n += len(yb)

print(f"\nAccuracy: {correct / n:.1%}")
print(f"Legendaries caught: {tp}/{tp + fn}   False alarms: {fp}")

# ------------------------------------------------------- 5. FUNC
def is_legendary(hp, atk, df_, spa, spd, spe):
    x = scaler.transform([[hp, atk, df_, spa, spd, spe]])
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).item()
    return p

# ------------------------------------------------------- 6. TRY YOUR OWN
print("\n--- Try your own Pokémon ---")
print("Enter stats as six numbers: HP Attack Defense SpAtk SpDef Speed")
print("Example: 100 100 100 100 100 100   (type 'q' to quit)\n")
 
while True:
    text = input("Stats> ").strip().replace(",", " ")
    if text.lower() in ("q", "quit", "exit"):
        print("Bye!")
        break
    try:
        stats = [float(v) for v in text.split()]
        if len(stats) != 6:
            raise ValueError
    except ValueError:
        print("Please enter exactly six numbers (or 'q' to quit).")
        continue
    p = is_legendary(*stats)
    verdict = "Legendary!" if p > 0.5 else "Not Legendary."
    print(f"  {p:.0%} chance Legendary -> {verdict}")