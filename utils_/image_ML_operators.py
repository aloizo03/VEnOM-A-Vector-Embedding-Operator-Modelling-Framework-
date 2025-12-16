import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        patience: how many epochs to wait before stopping
        min_delta: min improvement to count as "better"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def update(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True



def get_scheduler(optimizer, schedule_type="none"):
    if schedule_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    if schedule_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

    if schedule_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5
        )

    return None 

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = nn.Conv2d(1, 32, 3, padding=1)
        self.block1 = ResBlock(32)
        self.block2 = ResBlock(32)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.start(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_ch=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )  # → (B, embed_dim, H/ps, W/ps)

    def forward(self, x):
        x = self.proj(x)              # (B, embed_dim, n, n)
        x = x.flatten(2)              # (B, embed_dim, patches)
        x = x.transpose(1, 2)         # (B, patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.att(x, x, x)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=4):
        super().__init__()

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, embed_dim=64, depth=4, num_heads=4, num_classes=10):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        return self.head(cls_out)


def fit(model, train_data, train_y, val_data=None, val_y=None, use_val=False, val_fraction=0.2, batch_size=64, lr=1e-3, max_epochs=100, use_early_stopping=True, early_patience=5, scheduler_type="none"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, scheduler_type)

    early_stop = EarlyStopping(patience=early_patience) if use_early_stopping else None

    full_train_dataset = ImgDataset(train_data, train_y)

    if use_val:
        # User-provided val_data has priority
        if val_data is not None and val_y is not None:
            val_dataset = ImgDataset(val_data, val_y)
            train_dataset = full_train_dataset
        else:
            # Split train into train/val
            val_size = int(len(full_train_dataset) * val_fraction)
            train_size = len(full_train_dataset) - val_size
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    else:
        train_dataset = full_train_dataset
        val_dataset = None

    # ---- Dataloaders ----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, correct = 0, 0

        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---- Validation if enabled ----
        if use_val:
            model.eval()
            total_loss, correct = 0, 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    total_loss += loss.item()
                    correct += (pred.argmax(1) == y).sum().item()

            val_loss = total_loss / len(val_loader)
            val_acc = correct / len(val_loader.dataset)

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch}/{max_epochs} | "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

            # Scheduler update
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            elif scheduler:
                scheduler.step()

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")

            # Early stopping check
            if early_stop:
                early_stop.update(val_loss)
                if early_stop.should_stop:
                    print(f"EARLY STOPPING at epoch {epoch}")
                    break

        else:
            # No validation mode
            print(f"Epoch {epoch}/{max_epochs} | "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%")


    return model

