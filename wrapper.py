# %% Import Necessary Modules
import logging
import os
import pathlib
from abc import ABC
from argparse import ArgumentParser
from typing import Dict, List, Callable

import torch
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model

# %%
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# %%
class DiodePair(nn.Module):
    def __init__(self, num_layers: int = 4, layer_size: int = 32, input_size: int = 2, output_size: int = 1,
                 activation_fn: Callable = torch.relu):
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
        self.dp = torch.load("fine_tuned_models/diode_pair_1u1d_1N4148_l_1.70k_2.50k_3x16_9.91e-05_f_5.40e-05.pth")
        self.dp_r = None

        self.update_impedance()

        self.p1_a = torch.zeros(3, dtype=torch.float32)
        self.p1_b = torch.zeros(3, dtype=torch.float32)

    def update_impedance(self):
        # Update parameters of parallel adaptor
        self.p1_r = torch.stack([(self.c1_r * self.vs_r) / (self.c1_r + self.vs_r), self.c1_r, self.vs_r])
        self.p1_g = torch.reciprocal(self.p1_r)
        self.p1 = (2 / torch.sum(self.p1_g)) * torch.ones(3, 1) @ self.p1_g.unsqueeze(0) - torch.eye(3)

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


class NeuralDiodeClipperWrapper(WaveformToWaveformBase):
    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        super().__init__(model, use_debug_mode)

    def get_model_name(self) -> str:
        return "clipper"

    def get_model_authors(self) -> List[str]:
        return ["Shijie Yang"]

    def get_model_short_description(self) -> str:
        return "Neural soft clipper effect"

    def get_model_long_description(self) -> str:
        return "Neural soft clipper effect implemented by emulating RC diode clipper circuits"

    def get_technical_description(self) -> str:
        return "Neural soft clipper effect implemented by emulating RC diode clipper circuits with differentiable \
                wave digital filters. Base on the paper of Jatin Chowdhury and Christopher Johann Clarke"

    def get_tags(self) -> List[str]:
        return ["distortion", "clipper"]

    def get_model_version(self) -> str:
        return "0.0.2"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://www.researchgate.net/profile/Christopher-Clarke-13/publication"
                     "/361416911_Emulating_Diode_Circuits_with_Differentiable_Wave_Digital_Filters/links"
                     "/62b04cee23f3283e3af84099/Emulating-Diode-Circuits-with-Differentiable-Wave-Digital-Filters.pdf"
        }

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True

    def set_model_sample_rate_and_buffer_size(self, sample_rate: int, n_samples: int) -> bool:
        self.model.set_fs(torch.tensor(sample_rate, dtype=torch.float32))
        return True

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [44100, 48000]

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("vs_r", "Tone", default_value=0.5),
            NeutoneParameter("test", "test", default_value=0.5),
        ]

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        vs_r = 10e3 + (100e3 - 10e3) * params["vs_r"]
        x = self.model.forward(x.squeeze(), vs_r.squeeze())
        return x.unsqueeze(0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    model = NDC()
    wrapper = NeuralDiodeClipperWrapper(model)
    save_neutone_model(wrapper, root_dir, dump_samples=True, submission=True)
