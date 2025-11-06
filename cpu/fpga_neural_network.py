"""
FPGA Neural Network Inference Module

This module implements the neural network inference that is supposed to be done on the FPGA,
but in Python for testing and development purposes.

The network architecture consists of:
1. Input standardization (z-score normalization)
2. ConvStack: 3 convolutional layers with batch norm and activations
3. LSTMStack: 6 bidirectional LSTM layers (alternating reverse/forward)
4. Output: [1, 384, 1667] tensor of LSTM features

Based on the README.md specifications and model config.toml.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


def load_tensor_file(tensor_path: str) -> torch.Tensor:
    """
    Load a tensor from a .tensor file.
    
    Args:
        tensor_path: Path to the .tensor file
        
    Returns:
        Extracted weight tensor
    """
    tensor_path = Path(tensor_path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
    
    # Load tensor from file
    loaded_data = torch.load(tensor_path, map_location='cpu', weights_only=False)
    
    # Handle different return types from torch.load()
    if isinstance(loaded_data, list):
        if len(loaded_data) == 0:
            raise ValueError(f"Empty tensor list in file: {tensor_path}")
        weight_tensor = loaded_data[0]
    elif isinstance(loaded_data, torch.Tensor):
        weight_tensor = loaded_data
    elif hasattr(loaded_data, 'state_dict'):
        state_dict = loaded_data.state_dict()
        if len(state_dict) == 0:
            raise ValueError(f"Empty state_dict in TorchScript module: {tensor_path}")
        weight_tensor = next(iter(state_dict.values()))
        if not isinstance(weight_tensor, torch.Tensor):
            raise ValueError(f"Expected tensor in state_dict, got {type(weight_tensor)}")
    elif hasattr(loaded_data, 'parameters'):
        params = list(loaded_data.parameters())
        if len(params) == 0:
            raise ValueError(f"No parameters in module: {tensor_path}")
        weight_tensor = params[0]
    else:
        raise ValueError(f"Unexpected data type from {tensor_path}: {type(loaded_data)}")
    
    return weight_tensor


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class ConvLayer(nn.Module):
    """Convolutional layer with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: str = "swish",
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True  # Bias is always used according to config
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            # Batch norm without learnable parameters (affine=False)
            # In inference mode, this will use running statistics if available,
            # otherwise it will normalize based on current batch statistics
            # Since batch norm params aren't in model files, we'll use eval mode
            # which should work for inference
            self.bn = nn.BatchNorm1d(out_channels, affine=False, track_running_stats=False)
        
        if activation == "swish":
            self.activation = swish
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class LSTMLayer(nn.Module):
    """Unidirectional LSTM layer (forward or reverse)."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        reverse: bool = False
    ):
        super().__init__()
        self.reverse = reverse
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,  # (seq_len, batch, features)
            bidirectional=False,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, features, seq_len]
        # Convert to [seq_len, batch, features] for LSTM
        x = x.permute(2, 0, 1)  # [seq_len, batch, features]
        
        if self.reverse:
            x = torch.flip(x, dims=[0])
        
        output, (h_n, c_n) = self.lstm(x)
        
        if self.reverse:
            output = torch.flip(output, dims=[0])
        
        # Convert back to [batch, features, seq_len]
        output = output.permute(1, 2, 0)  # [batch, features, seq_len]
        
        return output


class FPGANeuralNetwork(nn.Module):
    """
    FPGA Neural Network for basecalling feature extraction.
    
    This implements the feature extraction portion of the CRF neural network:
    - Input: Raw signal data (10,000 samples, 16-bit floats)
    - Output: LSTM features of shape [1, 384, 1667]
    """
    
    def __init__(
        self,
        model_dir: str,
        standardize_mean: float = 94.0,
        standardize_stdev: float = 24.0
    ):
        super().__init__()
        
        self.standardize_mean = standardize_mean
        self.standardize_stdev = standardize_stdev
        self.model_dir = Path(model_dir)
        
        # ConvStack: 3 convolutional layers
        # Conv1: 1→16 channels, kernel=5, stride=1, padding=2, activation=swish, batch_norm
        self.conv1 = ConvLayer(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            activation="swish",
            use_batch_norm=True
        )
        
        # Conv2: 16→16 channels, kernel=5, stride=1, padding=2, activation=swish, batch_norm
        self.conv2 = ConvLayer(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            activation="swish",
            use_batch_norm=True
        )
        
        # Conv3: 16→384 channels, kernel=19, stride=6, padding=9, activation=tanh, batch_norm
        self.conv3 = ConvLayer(
            in_channels=16,
            out_channels=384,
            kernel_size=19,
            stride=6,
            padding=9,
            activation="tanh",
            use_batch_norm=True
        )
        
        # LSTMStack: 6 bidirectional LSTM layers
        # Alternating directions: [reverse, forward, reverse, forward, reverse, forward]
        self.lstm1 = LSTMLayer(input_size=384, hidden_size=384, reverse=True)
        self.lstm2 = LSTMLayer(input_size=384, hidden_size=384, reverse=False)
        self.lstm3 = LSTMLayer(input_size=384, hidden_size=384, reverse=True)
        self.lstm4 = LSTMLayer(input_size=384, hidden_size=384, reverse=False)
        self.lstm5 = LSTMLayer(input_size=384, hidden_size=384, reverse=True)
        self.lstm6 = LSTMLayer(input_size=384, hidden_size=384, reverse=False)
        
        # Load weights from model directory
        self.load_weights()
    
    def load_weights(self):
        """Load weights from .tensor files in the model directory."""
        print(f"Loading weights from {self.model_dir}")
        
        # Load convolutional layer weights
        # Conv1: 0.conv.weight.tensor, 0.conv.bias.tensor
        conv1_weight = load_tensor_file(str(self.model_dir / "0.conv.weight.tensor"))
        conv1_bias = load_tensor_file(str(self.model_dir / "0.conv.bias.tensor"))
        self.conv1.conv.weight.data = conv1_weight
        self.conv1.conv.bias.data = conv1_bias
        print(f"  Loaded Conv1: weight {conv1_weight.shape}, bias {conv1_bias.shape}")
        
        # Conv2: 1.conv.weight.tensor, 1.conv.bias.tensor
        conv2_weight = load_tensor_file(str(self.model_dir / "1.conv.weight.tensor"))
        conv2_bias = load_tensor_file(str(self.model_dir / "1.conv.bias.tensor"))
        self.conv2.conv.weight.data = conv2_weight
        self.conv2.conv.bias.data = conv2_bias
        print(f"  Loaded Conv2: weight {conv2_weight.shape}, bias {conv2_bias.shape}")
        
        # Conv3: 2.conv.weight.tensor, 2.conv.bias.tensor
        conv3_weight = load_tensor_file(str(self.model_dir / "2.conv.weight.tensor"))
        conv3_bias = load_tensor_file(str(self.model_dir / "2.conv.bias.tensor"))
        self.conv3.conv.weight.data = conv3_weight
        self.conv3.conv.bias.data = conv3_bias
        print(f"  Loaded Conv3: weight {conv3_weight.shape}, bias {conv3_bias.shape}")
        
        # Load LSTM layer weights
        # LSTM layers are numbered 4, 5, 6, 7, 8, 9 (skipping 3 which might be permute)
        # Each LSTM has: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        lstm_layers = [
            (self.lstm1, 4),
            (self.lstm2, 5),
            (self.lstm3, 6),
            (self.lstm4, 7),
            (self.lstm5, 8),
            (self.lstm6, 9),
        ]
        
        for lstm_layer, layer_idx in lstm_layers:
            weight_ih = load_tensor_file(str(self.model_dir / f"{layer_idx}.rnn.weight_ih_l0.tensor"))
            weight_hh = load_tensor_file(str(self.model_dir / f"{layer_idx}.rnn.weight_hh_l0.tensor"))
            bias_ih = load_tensor_file(str(self.model_dir / f"{layer_idx}.rnn.bias_ih_l0.tensor"))
            bias_hh = load_tensor_file(str(self.model_dir / f"{layer_idx}.rnn.bias_hh_l0.tensor"))
            
            # PyTorch LSTM expects weights in a specific format
            # Combine input-hidden and hidden-hidden weights
            # Shape: [4*hidden_size, input_size] for weight_ih
            # Shape: [4*hidden_size, hidden_size] for weight_hh
            lstm_layer.lstm.weight_ih_l0.data = weight_ih
            lstm_layer.lstm.weight_hh_l0.data = weight_hh
            lstm_layer.lstm.bias_ih_l0.data = bias_ih
            lstm_layer.lstm.bias_hh_l0.data = bias_hh
            
            print(f"  Loaded LSTM{layer_idx}: weight_ih {weight_ih.shape}, weight_hh {weight_hh.shape}")
        
        print("Weights loaded successfully!")
    
    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization.
        
        Args:
            x: Input signal tensor of shape [batch, 1, time] or [batch, time]
            
        Returns:
            Standardized tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Standardize: (x - mean) / stdev
        x = (x - self.standardize_mean) / self.standardize_stdev
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        
        Args:
            x: Input signal tensor of shape [batch, time] or [batch, 1, time]
               Values should be 16-bit floats (normalized signal)
               Expected length: 10,000 samples
               
        Returns:
            LSTM features of shape [batch, 384, 1667]
        """
        # Standardize input
        x = self.standardize(x)
        
        # Ensure correct shape: [batch, channels, time]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, time]
        
        # ConvStack
        x = self.conv1(x)  # [batch, 16, 10000]
        x = self.conv2(x)  # [batch, 16, 10000]
        x = self.conv3(x)  # [batch, 384, 1667] (stride=6 reduces time dimension)
        
        # LSTMStack
        x = self.lstm1(x)  # [batch, 384, 1667]
        x = self.lstm2(x)  # [batch, 384, 1667]
        x = self.lstm3(x)  # [batch, 384, 1667]
        x = self.lstm4(x)  # [batch, 384, 1667]
        x = self.lstm5(x)  # [batch, 384, 1667]
        x = self.lstm6(x)  # [batch, 384, 1667]
        
        return x


def create_fpga_network(model_dir: str) -> FPGANeuralNetwork:
    """
    Create and initialize FPGA neural network from model directory.
    
    Args:
        model_dir: Path to model directory containing .tensor files
        
    Returns:
        Initialized FPGANeuralNetwork module
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    network = FPGANeuralNetwork(model_dir=str(model_dir))
    network.eval()  # Set to evaluation mode
    
    return network

