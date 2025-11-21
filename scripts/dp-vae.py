from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_PATH = Path("../data/telemetry.csv")
OUTPUT_PATH = Path("../data/synthetic.csv")
BATCH_SIZE = 256
LATENT_DIMENSION = 16
HIDDEN_SIZE = 128
EPOCHS = 30
NOISE_MULTIPLIER = 1.0
MAXIMUM_GRADIENT_NORM = 1.0
TARGET_EPSILON = 4.0
DELTA = 1e-5
SYNTHETIC_SAMPLES = 10_000


class TelemetryDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        categorical_targets: torch.Tensor,
        numeric_targets: torch.Tensor,
    ) -> None:
        self.features = features
        self.categorical_targets = categorical_targets
        self.numeric_targets = numeric_targets

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.categorical_targets[index],
            self.numeric_targets[index],
        )


class TelemetryVAE(nn.Module):
    def __init__(
        self, input_dimension: int, latent_dimension: int, categorical_sizes: List[int]
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(HIDDEN_SIZE, latent_dimension)
        self.logvar_layer = nn.Linear(HIDDEN_SIZE, latent_dimension)

        self.categorical_decoders = nn.ModuleList(
            [nn.Linear(latent_dimension, size) for size in categorical_sizes]
        )
        self.numeric_decoder = nn.Linear(latent_dimension, 1)

    def encode(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_tensor)
        return self.mean_layer(hidden), self.logvar_layer(hidden)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, latent_tensor: torch.Tensor):
        categorical_logits = [decoder(latent_tensor)
                              for decoder in self.categorical_decoders]
        numeric_output = self.numeric_decoder(latent_tensor)
        return categorical_logits, numeric_output

    def forward(self, input_tensor: torch.Tensor):
        mean, logvar = self.encode(input_tensor)
        latent_tensor = self.reparameterize(mean, logvar)
        categorical_logits, numeric_output = self.decode(latent_tensor)
        return categorical_logits, numeric_output, mean, logvar


def load_and_preprocess():
    dataframe = pd.read_csv(DATA_PATH)
    if "User ID" in dataframe.columns:
        dataframe = dataframe.drop(columns=["User ID"])

    dataframe["TimeSeconds"] = (
        pd.to_datetime(dataframe["Time of Event"]).astype(
            "int64") // 1_000_000_000
    )

    categorical_columns = ["Product Type", "Event Type"]
    numeric_columns = ["TimeSeconds"]

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", StandardScaler(), numeric_columns),
        ]
    )

    feature_matrix = transformer.fit_transform(
        dataframe[categorical_columns + numeric_columns]
    )
    categorical_encoder: OneHotEncoder = transformer.named_transformers_[
        "categorical"]
    categorical_sizes: List[int] = [
        len(category) for category in categorical_encoder.categories_
    ]
    categorical_total = sum(categorical_sizes)
    numeric_scaler: StandardScaler = transformer.named_transformers_["numeric"]

    categorical_target_list = []
    start_index = 0
    for size in categorical_sizes:
        category_block = feature_matrix[:, start_index: start_index + size]
        categorical_target_list.append(category_block.argmax(axis=1))
        start_index += size

    categorical_targets = np.stack(
        categorical_target_list, axis=1).astype(np.int64)
    numeric_targets = feature_matrix[:, categorical_total:].astype(np.float32)

    features = torch.tensor(feature_matrix, dtype=torch.float32)
    categorical_target_tensor = torch.tensor(
        categorical_targets, dtype=torch.long)
    numeric_target_tensor = torch.tensor(numeric_targets, dtype=torch.float32)

    dataset = TelemetryDataset(
        features, categorical_target_tensor, numeric_target_tensor
    )

    time_minimum, time_maximum = (
        dataframe["TimeSeconds"].min(),
        dataframe["TimeSeconds"].max(),
    )
    return (
        dataframe,
        dataset,
        time_minimum,
        time_maximum,
        categorical_columns,
        numeric_columns,
        categorical_sizes,
        numeric_scaler,
        categorical_encoder,
    )


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()


def train_model(
    dataset: TelemetryDataset,
    input_dimension: int,
    categorical_sizes: List[int],
    device: torch.device,
):
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    model = TelemetryVAE(
        input_dimension=input_dimension,
        latent_dimension=LATENT_DIMENSION,
        categorical_sizes=categorical_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    categorical_loss = nn.CrossEntropyLoss()
    numeric_loss = nn.MSELoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAXIMUM_GRADIENT_NORM,
    )

    epsilon_history: List[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_features, batch_categorical_targets, batch_numeric_targets in dataloader:
            batch_features = batch_features.to(device)
            batch_categorical_targets = batch_categorical_targets.to(device)
            batch_numeric_targets = batch_numeric_targets.to(device)

            optimizer.zero_grad()
            categorical_logits, numeric_prediction, mean, logvar = model(
                batch_features)

            categorical_loss_value = sum(
                categorical_loss(logits, batch_categorical_targets[:, index])
                for index, logits in enumerate(categorical_logits)
            )
            numeric_loss_value = numeric_loss(
                numeric_prediction, batch_numeric_targets
            )
            kl_loss = kl_divergence(mean, logvar)

            loss = categorical_loss_value + numeric_loss_value + kl_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_features.size(0)

        epoch_loss = running_loss / len(dataset)
        try:
            epsilon = privacy_engine.accountant.get_epsilon(delta=DELTA)
        except ValueError:
            epsilon = float("nan")

        epsilon_history.append(epsilon)
        print(f"Epoch {epoch:02d}: loss={epoch_loss:.4f}, ε={epsilon:.3f}")

        if (
            np.isfinite(TARGET_EPSILON)
            and np.isfinite(epsilon)
            and epsilon >= TARGET_EPSILON
        ):
            print("TARGET_EPSILON reached. Stopping early!")
            break

    return model, epsilon_history


def sample_synthetic(
    model: nn.Module,
    device: torch.device,
    categorical_columns: List[str],
    numeric_columns: List[str],
    numeric_scaler: StandardScaler,
    categorical_encoder: OneHotEncoder,
    time_minimum: int,
    time_maximum: int,
):
    inference_model = model._module if hasattr(model, "_module") else model
    inference_model.eval()
    latent_samples = torch.randn(
        SYNTHETIC_SAMPLES, LATENT_DIMENSION, device=device)

    with torch.no_grad():
        categorical_logits, numeric_prediction = inference_model.decode(
            latent_samples)

    decoded_categories = []
    for column_name, logits, labels in zip(
        categorical_columns, categorical_logits, categorical_encoder.categories_
    ):
        label_indices = logits.argmax(dim=1).cpu().numpy()
        decoded_categories.append(
            pd.Series(labels[label_indices], name=column_name)
        )

    categorical_dataframe = pd.concat(decoded_categories, axis=1)

    numeric_array = numeric_prediction.cpu().numpy()
    numeric_values = numeric_scaler.inverse_transform(numeric_array)
    numeric_dataframe = pd.DataFrame(numeric_values, columns=numeric_columns)
    numeric_dataframe["TimeSeconds"] = numeric_dataframe["TimeSeconds"].clip(
        time_minimum, time_maximum
    ).astype(np.int64)
    numeric_dataframe["Time of Event"] = pd.to_datetime(
        numeric_dataframe["TimeSeconds"], unit="s"
    )

    synthetic_dataframe = pd.concat(
        [
            categorical_dataframe.reset_index(drop=True),
            numeric_dataframe.reset_index(drop=True),
        ],
        axis=1,
    )
    synthetic_dataframe = synthetic_dataframe[
        categorical_columns + numeric_columns + ["Time of Event"]
    ]
    synthetic_dataframe.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH.resolve()}!")

    return synthetic_dataframe


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        dataframe,
        dataset,
        time_minimum,
        time_maximum,
        categorical_columns,
        numeric_columns,
        categorical_sizes,
        numeric_scaler,
        categorical_encoder,
    ) = load_and_preprocess()

    model, epsilon_history = train_model(
        dataset=dataset,
        input_dimension=dataset.features.shape[1],
        categorical_sizes=categorical_sizes,
        device=device,
    )

    sample_synthetic(
        model=model,
        device=device,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        numeric_scaler=numeric_scaler,
        categorical_encoder=categorical_encoder,
        time_minimum=time_minimum,
        time_maximum=time_maximum,
    )


if __name__ == "__main__":
    main()
