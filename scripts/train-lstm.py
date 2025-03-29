import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import DataLoader, random_split
import polars as pl
from tqdm import trange
from torchmetrics.classification import BinaryAUROC
from torchmetrics.aggregation import MeanMetric

from learned_fall_detection.dataset import FallenDataset
from learned_fall_detection.data_loading import load


def repack_like(x: torch.Tensor, like: PackedSequence) -> PackedSequence:
    return PackedSequence(
        x,
        batch_sizes=like.batch_sizes,
        sorted_indices=like.sorted_indices,
        unsorted_indices=like.unsorted_indices,
    )


class PredictionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size, proj_size=output_size)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: PackedSequence) -> PackedSequence:
        # lstm_out, _ = self.lstm(x)
        y = self.model(x.data)
        return repack_like(y, x)


def loss_fn(outputs: PackedSequence, targets: PackedSequence) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        outputs.data.view(-1),
        targets.data.float(),
    )


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[PackedSequence, PackedSequence]:
    features, targets = zip(*batch)

    packed_features = pack_sequence(features, enforce_sorted=False)
    packed_targets = pack_sequence(targets, enforce_sorted=False)

    return packed_features, packed_targets


def main(
    model: PredictionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in (pbar := trange(100)):
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_description_str(f"Train Epoch {epoch}, Loss: {loss.item()}")

        model.eval()
        with torch.inference_mode():
            auroc = BinaryAUROC().to(device)
            mean_loss = MeanMetric().to(device)
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                auroc.update(outputs.data, targets.data)
                loss = loss_fn(outputs, targets)
                mean_loss.update(loss)

            auroc_value = auroc.compute()
            mean_loss_value = mean_loss.compute()
            pbar.set_description_str(f"Val Epoch {epoch}, Loss: {mean_loss_value}, AUROC: {auroc_value}")


if __name__ == "__main__":
    df = load("data.parquet")
    dataset = FallenDataset(
        df,
        group_keys=["robot_identifier", "match_identifier"],
        features=[
            pl.col("Control.main_outputs.robot_orientation.pitch"),
            pl.col("Control.main_outputs.robot_orientation.roll"),
            pl.col("Control.main_outputs.robot_orientation.yaw"),
        ],
    )
    n_features = dataset.n_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PredictionModel(input_size=n_features, hidden_size=32, output_size=1).to(
        device
    )
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    main(model, train_loader, val_loader, device)
