#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Conditional imports for Flower
try:
    import flwr as fl
    from flwr.client import Client, NumPyClient
    from flwr.common import (
        Code,
        FitRes,
        GetParametersRes,
        GetPropertiesRes,
        Parameters,
        Scalar,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    logging.warning("Flower not installed. Install with: pip install flwr")

# Conditional imports for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. Install with: pip install torch")

from src.mobile.pqc_transport import PQCTransportLayer

logger = logging.getLogger(__name__)

@dataclass
class MobileDeviceConfig:
    
    
    client_id: str = "mobile_client_001"
    server_address: str = "localhost:8080"
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    enable_pqc: bool = True
    compress_gradients: bool = True
    compression_ratio: float = 0.1
    max_memory_mb: int = 512
    battery_aware: bool = True

class SimpleMobileModel(nn.Module):
    
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 10,
    ) -> None:
        
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)

class MobileFlowerClient:
    
    
    def __init__(
        self,
        config: MobileDeviceConfig,
        model: Optional[nn.Module] = None,
    ) -> None:
        
        self.config = config
        
        # Initialize model
        if model is not None:
            self.model = model
        elif TORCH_AVAILABLE:
            self.model = SimpleMobileModel()
        else:
            self.model = None
            logger.warning("PyTorch not available, model training disabled")
        
        # Initialize PQC transport
        if config.enable_pqc:
            self.pqc_transport = PQCTransportLayer(client_id=config.client_id)
        else:
            self.pqc_transport = None
        
        # Training state
        self._local_data: Optional[Tuple[NDArray, NDArray]] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._criterion = nn.CrossEntropyLoss() if TORCH_AVAILABLE else None
        
        # Metrics tracking
        self.training_history: List[Dict[str, Any]] = []
        self._round_number = 0
        
        logger.info(f"Initialized MobileFlowerClient: {config.client_id}")
    
    def set_local_data(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        validation_split: float = 0.1,
    ) -> None:
        
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        
        # Split
        self._local_data = (X[n_val:], y[n_val:])
        self._val_data = (X[:n_val], y[:n_val]) if n_val > 0 else None
        
        logger.info(
            f"Set local data: {len(self._local_data[0])} train, "
            f"{n_val} validation samples"
        )
    
    def get_parameters(self) -> List[NDArray]:
        
        if self.model is None:
            return []
        
        return [
            param.detach().cpu().numpy()
            for param in self.model.parameters()
        ]
    
    def set_parameters(self, parameters: List[NDArray]) -> None:
        
        if self.model is None:
            return
        
        for param, new_value in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(
                new_value,
                dtype=param.dtype,
                device=param.device,
            )
    
    def _create_data_loader(
        self,
        X: NDArray,
        y: NDArray,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")
        
        batch_size = batch_size or self.config.batch_size
        
        # Adapt batch size to memory constraints
        if self.config.battery_aware:
            batch_size = min(batch_size, self.config.max_memory_mb // 4)
        
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y),
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
    
    def train_local(
        self,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        
        if self.model is None or self._local_data is None:
            return {"error": "Model or data not available"}
        
        epochs = epochs or self.config.local_epochs
        X_train, y_train = self._local_data
        
        # Create optimizer if not exists
        if self._optimizer is None:
            self._optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        
        # Create data loader
        train_loader = self._create_data_loader(X_train, y_train)
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                self._optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self._criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self._optimizer.step()
                
                epoch_loss += loss.item() * len(batch_X)
                epoch_samples += len(batch_X)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            logger.debug(
                f"Epoch {epoch + 1}/{epochs}: "
                f"loss = {epoch_loss / epoch_samples:.4f}"
            )
        
        training_time = time.time() - start_time
        
        # Compute validation metrics if available
        val_metrics = {}
        if self._val_data is not None:
            val_metrics = self._evaluate(self._val_data[0], self._val_data[1])
        
        metrics = {
            "train_loss": total_loss / total_samples,
            "train_samples": total_samples,
            "epochs": epochs,
            "training_time_seconds": training_time,
            **val_metrics,
        }
        
        self.training_history.append({
            "round": self._round_number,
            **metrics,
        })
        
        return metrics
    
    def _evaluate(
        self,
        X: NDArray,
        y: NDArray,
    ) -> Dict[str, float]:
        
        if self.model is None:
            return {}
        
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            outputs = self.model(X_tensor)
            loss = self._criterion(outputs, y_tensor)
            
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == y_tensor).float().mean()
        
        self.model.train()
        
        return {
            "val_loss": loss.item(),
            "val_accuracy": accuracy.item(),
        }
    
    def compress_gradients(
        self,
        parameters: List[NDArray],
    ) -> List[NDArray]:
        
        if not self.config.compress_gradients:
            return parameters
        
        compressed = []
        ratio = self.config.compression_ratio
        
        for param in parameters:
            flat = param.flatten()
            k = max(1, int(len(flat) * ratio))
            
            # Top-K indices
            indices = np.argpartition(np.abs(flat), -k)[-k:]
            
            # Create sparse representation
            sparse = np.zeros_like(flat)
            sparse[indices] = flat[indices]
            
            compressed.append(sparse.reshape(param.shape))
        
        return compressed
    
    def encrypt_weights(
        self,
        parameters: List[NDArray],
    ) -> bytes:
        
        if self.pqc_transport is None:
            # Fallback: serialize without encryption
            import pickle
            return pickle.dumps(parameters)
        
        return self.pqc_transport.encrypt_weights(parameters)
    
    def sign_update(
        self,
        data: bytes,
    ) -> bytes:
        
        if self.pqc_transport is None:
            # Fallback: HMAC signature
            import hashlib
            import hmac
            return hmac.new(
                self.config.client_id.encode(),
                data,
                hashlib.sha256,
            ).digest()
        
        return self.pqc_transport.sign(data)
    
    def start(self) -> None:
        
        if not FLOWER_AVAILABLE:
            logger.error("Flower not available. Cannot start FL client.")
            return
        
        logger.info(
            f"Starting FL client {self.config.client_id} "
            f"connecting to {self.config.server_address}"
        )
        
        # Create Flower client adapter
        flower_client = FlowerClientAdapter(self)
        
        # Start Flower client
        fl.client.start_client(
            server_address=self.config.server_address,
            client=flower_client.to_client(),
        )

class FlowerClientAdapter(NumPyClient):
    
    
    def __init__(self, mobile_client: MobileFlowerClient) -> None:
        
        self.mobile_client = mobile_client
    
    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        
        return {
            "client_id": self.mobile_client.config.client_id,
            "pqc_enabled": str(self.mobile_client.config.enable_pqc),
            "battery_aware": str(self.mobile_client.config.battery_aware),
        }
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[NDArray]:
        
        return self.mobile_client.get_parameters()
    
    def fit(
        self,
        parameters: List[NDArray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[NDArray], int, Dict[str, Scalar]]:
        
        # Set global parameters
        self.mobile_client.set_parameters(parameters)
        
        # Increment round number
        self.mobile_client._round_number += 1
        
        # Get training config from server
        local_epochs = int(config.get("local_epochs", self.mobile_client.config.local_epochs))
        
        # Train locally
        metrics = self.mobile_client.train_local(epochs=local_epochs)
        
        # Get updated parameters
        updated_params = self.mobile_client.get_parameters()
        
        # Apply compression if enabled
        if self.mobile_client.config.compress_gradients:
            updated_params = self.mobile_client.compress_gradients(updated_params)
        
        # Get sample count
        n_samples = self.mobile_client._local_data[0].shape[0] if self.mobile_client._local_data else 0
        
        return updated_params, n_samples, metrics
    
    def evaluate(
        self,
        parameters: List[NDArray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        
        self.mobile_client.set_parameters(parameters)
        
        if self.mobile_client._val_data is None:
            return 0.0, 0, {}
        
        X_val, y_val = self.mobile_client._val_data
        metrics = self.mobile_client._evaluate(X_val, y_val)
        
        return (
            metrics.get("val_loss", 0.0),
            len(X_val),
            {"accuracy": metrics.get("val_accuracy", 0.0)},
        )

def run_simulation(
    n_clients: int = 5,
    n_rounds: int = 10,
    samples_per_client: int = 1000,
) -> None:
    
    logger.info(
        f"Starting FL simulation: {n_clients} clients, "
        f"{n_rounds} rounds, {samples_per_client} samples/client"
    )
    
    # Create clients
    clients = []
    for i in range(n_clients):
        config = MobileDeviceConfig(
            client_id=f"mobile_{i:03d}",
            enable_pqc=True,
        )
        client = MobileFlowerClient(config)
        
        # Generate synthetic data (non-IID simulation)
        # Each client gets data with slight distribution shift
        X = np.random.randn(samples_per_client, 784)
        y = np.random.randint(0, 10, samples_per_client)
        
        # Add client-specific bias
        X += i * 0.1
        
        client.set_local_data(X, y)
        clients.append(client)
    
    # Initialize global parameters from first client
    global_params = clients[0].get_parameters()
    
    # Simulation loop
    for round_num in range(n_rounds):
        logger.info(f"=== Round {round_num + 1}/{n_rounds} ===")
        
        # Collect updates from all clients
        all_updates = []
        total_samples = 0
        
        for client in clients:
            client._round_number = round_num
            client.set_parameters([p.copy() for p in global_params])
            
            # Local training
            metrics = client.train_local()
            
            # Get compressed updates
            updates = client.get_parameters()
            if client.config.compress_gradients:
                updates = client.compress_gradients(updates)
            
            n_samples = client._local_data[0].shape[0]
            all_updates.append((updates, n_samples))
            total_samples += n_samples
            
            logger.info(
                f"  {client.config.client_id}: "
                f"loss={metrics.get('train_loss', 0):.4f}, "
                f"samples={n_samples}"
            )
        
        # Aggregate updates (FedAvg)
        new_global_params = []
        for param_idx in range(len(global_params)):
            weighted_sum = sum(
                updates[param_idx] * n_samples
                for updates, n_samples in all_updates
            )
            new_global_params.append(weighted_sum / total_samples)
        
        global_params = new_global_params
        
        # Compute global metrics
        avg_loss = np.mean([
            client.training_history[-1].get("train_loss", 0)
            for client in clients
        ])
        logger.info(f"  Global average loss: {avg_loss:.4f}")
    
    logger.info("Simulation complete!")

def main():
    
    parser = argparse.ArgumentParser(
        description="Q-Edge Mobile Federated Learning Client"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:8080",
        help="FL server address",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default="mobile_001",
        help="Client identifier",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run local simulation instead of connecting to server",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=5,
        help="Number of clients for simulation",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=10,
        help="Number of FL rounds",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if args.simulate:
        run_simulation(
            n_clients=args.n_clients,
            n_rounds=args.n_rounds,
        )
    else:
        config = MobileDeviceConfig(
            client_id=args.client_id,
            server_address=args.server,
        )
        
        client = MobileFlowerClient(config)
        
        # Generate sample data
        X = np.random.randn(1000, 784)
        y = np.random.randint(0, 10, 1000)
        client.set_local_data(X, y)
        
        client.start()

if __name__ == "__main__":
    main()
