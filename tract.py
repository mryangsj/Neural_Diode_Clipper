import torch
import torch.nn as nn
from typing import Dict, List, Callable


class DiodePair(nn.Module):
    def __init__(self, num_layers: int, layer_size: int, input_size: int = 2, output_size: int = 1,
                 activation_fn: Callable = nn.ELU()):
        """
        A fully connected neural network module of WDF diode pair .

        Args:
            num_layers: The number of layers in the network.
            layer_size: The size of each layer.
            input_size: The size of the input layer. Defaults to 2.
            output_size: The size of the output layer. Defaults to 1.
            activation_fn: The activation function to use. Defaults to ReLU.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.activation_fn = activation_fn

        self.input_layer = nn.Linear(input_size, layer_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_size, layer_size) for _ in range(num_layers-1)])
        self.output_layer = nn.Linear(layer_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        y = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            y = self.activation_fn(hidden_layer(y))
        y = self.output_layer(y)
        return y


class NDC(nn.Module):
    def __init__(self, fs: float = 48e3, vs_r: float = 10e3, c1: float = 4.7e-9,
                 dp_path=None, num_layers: int = 3, layer_size: int = 16):
        super().__init__()
        self.fs = torch.tensor(fs, dtype=torch.float32)
        self.c1 = torch.tensor(c1, dtype=torch.float32)

        self.vs_r = torch.tensor(vs_r, dtype=torch.float32)
        self.c1_r = 1 / (2 * self.c1 * self.fs)
        self.p1_r = None
        self.p1_g = None
        self.p1 = None

        # self.dp = torch.load(dp_path) if dp_path else DiodePair(num_layers=num_layers, layer_size=layer_size)
        self.dp = torch.load("fine_tuned_models/diode_pair_1u1d_1N4148_l_1.70k_2.50k_2x8_1.40e-04_f_5.69e-05x.pth")
        self.dp_r = None

        self.update_impedance()

        self.p1_a = torch.zeros(3, dtype=torch.float32)
        self.p1_b = torch.zeros(3, dtype=torch.float32)

    def update_impedance(self) -> None:
        # Update parameters of parallel adaptor
        self.p1_r = torch.stack([(self.c1_r * self.vs_r) / (self.c1_r + self.vs_r), self.c1_r, self.vs_r])
        self.p1_g = torch.reciprocal(self.p1_r)
        self.p1 = (2 / torch.sum(self.p1_g)) * torch.ones(3, 1) @ self.p1_g.unsqueeze(0) - torch.eye(3)

        self.test = torch.ones(3, 1) @ self.p1_g.unsqueeze(0)

        # Update parameter of input impedance of diode pair
        self.dp_r = (self.p1_r[0] - 1.7e3) / (2.5e3 - 1.7e3)

    def set_fs(self, fs: torch.Tensor) -> None:
        if self.fs == fs:
            return
        else:
            self.fs = fs
            self.c1_r = 1 / (2 * self.c1 * self.fs)
            self.update_impedance()

    def set_vs_r(self, vs_r: torch.Tensor) -> None:
        if self.vs_r == vs_r:
            return
        else:
            self.vs_r = vs_r
            self.update_impedance()

    def forward(self, v_in: torch.Tensor, vs_r: torch.Tensor) -> torch.Tensor:
        # states initiate
        self.set_vs_r(vs_r)

        # waves initiate
        output = torch.zeros_like(v_in)

        for i in range(len(v_in)):
            # Forward Scan
            self.p1_a[1] = self.p1_b[1]
            self.p1_a[2] = v_in[i]
            self.p1_b = self.p1 @ self.p1_a

            # Local Root Scattering
            dp_a = self.p1_b[0]
            dp_b = self.dp(torch.stack([dp_a.squeeze(), self.dp_r])).squeeze(0)

            # Backward Scan
            self.p1_a[0] = dp_b
            self.p1_b = self.p1 @ self.p1_a

            # Read Output
            output[i] = (dp_a + dp_b) / 2

        return output

# model = NDC()
model = torch.load("./pre_trained_models/1u2d/diode_pair_1u2d_1N4148_l_1.70k_2.50k_2x8_1.91e-06.pth")
traced_net = torch.jit.trace(model, torch.rand(1024, 2))
torch.jit.save(traced_net, "1n34a_1u2d_2x8.pt")
