# %% Import Necessary Modules
import torch
import time
import librosa
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable

# %% Check Device
device = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} as backend.")


# %% Prepare Train and Validate Data
def diodeClipper_dataset_load(N, Fs, num_up, num_down, r_begin, r_end):
    r_begin_str = f"{r_begin / 1000:.2f}k"
    r_end_str = f"{r_end / 1000:.2f}k"
    Fs_str = f"{Fs / 1000:.1f}k"

    data_path = ("./LTspice_data" +
                 f"/{num_up}u{num_down}d" + "/diode_clipper" + f"_{num_up}u{num_down}d" +
                 f"_{r_begin_str}" + f"_{r_end_str}" + f"_{Fs_str}"
                 )

    v_i_path = data_path + "_i.wav"
    v_o_path = data_path + "_o.wav"
    r_path = data_path + "_r.wav"

    v_i, _ = librosa.load(path=v_i_path, sr=None, mono=True)
    v_o, _ = librosa.load(path=v_o_path, sr=None, mono=True)
    r, _ = librosa.load(path=r_path, sr=None, mono=True)

    if len(v_i) != len(v_o):
        raise ValueError("The length of v_in and v_out are not same!")
    if len(v_i) != len(r):
        raise ValueError("The length of v_in and r are not same!")

    if N <= len(v_i):
        v_i = v_i[:N]
        v_o = v_o[:N]
        r   = r  [:N]
    else:
        raise ValueError("The N must be less or equal than the length of audio files.")

    r = r_begin + (r_end - r_begin) * r
    Fs = np.ones_like(v_i) * Fs
    return np.stack((v_i, r, Fs), axis=-1), v_o


def get_diodeClipper_dataloader(N, Fs=None, num_up=1, num_down=1, r_begin: float = 10e3, r_end: float = 100e3,
                                batch_size=16, split_ratio=0.8, shuffle=True, random_state=42, plot=False):
    if Fs is None:
        Fs = [48e3]

    x, y = [], []
    for Fs in Fs:
        v_i_r, v_o = diodeClipper_dataset_load(N, Fs, num_up, num_down, r_begin, r_end)
        x.extend(v_i_r)
        y.extend(v_o)
    x = np.array(x).squeeze()
    y = np.array(y).squeeze()

    num_batches = int(len(x) / batch_size)
    x_batched = x[:num_batches * batch_size].reshape(num_batches, batch_size, 3)
    y_batched = y[:num_batches * batch_size].reshape(num_batches, batch_size)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_batched, y_batched,
        train_size=split_ratio,
        shuffle=shuffle,
        random_state=random_state
    )

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device),
    )
    valid_ds = TensorDataset(
        torch.tensor(x_valid, dtype=torch.float32).to(device),
        torch.tensor(y_valid, dtype=torch.float32).to(device)
    )

    train_dl = DataLoader(train_ds, shuffle=True)
    valid_dl = DataLoader(valid_ds)

    if plot:
        v_in = x[:, 0]
        r_in = x[:, 1]
        v_out = y

        # fig_size = (18, 7)
        fig, ax1_1 = plt.subplots()

        line1, = ax1_1.plot(v_in, label='v_in')
        line2, = ax1_1.plot(v_out, label='v_out')
        ax1_1.set_ylim((-1.0, 1.0))
        ax1_1.set_ylabel('Voltage [V]')

        ax1_2 = ax1_1.twinx()
        line3, = ax1_2.plot(r_in, color='g', label='R_in')
        ax1_2.set_ylabel('Resistor [Ω]')
        ax1_2.set_ylim((r_begin, r_end))
        ax1_2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]
        ax1_1.legend(lines, labels, loc='upper right')

        plt.show()

    return train_dl, valid_dl


# %% Neural Diode Clipper
class DiodePair(nn.Module):
    def __init__(self, num_layers: int = 4, layer_size: int = 32, input_size: int = 2, output_size: int = 1, activation_fn: Callable = torch.relu):
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
    def __init__(self, fs: float = 48e3, c1: float = 4.7e-9, vs_r: float = 10e3,
                 dp_path=None, num_layers: int = 2, layer_size: int = 16):
        super().__init__()
        self.fs = torch.tensor(fs, dtype=torch.float32)
        self.c1 = torch.tensor(c1, dtype=torch.float32)

        self.vs_r = torch.tensor(vs_r, dtype=torch.float32)
        self.c1_r = None
        self.p1_r = None
        self.p1_g = None
        self.p1 = None

        self.dp = torch.load(dp_path) if dp_path else DiodePair(num_layers=num_layers, layer_size=layer_size)
        self.dp_r = None

        self.update_parameter()

        self.dp = torch.load(dp_path) if dp_path else DiodePair(num_layers=num_layers, layer_size=layer_size)

    def set_states(self, vs_r, fs):
        self.vs_r = vs_r
        self.fs = fs
        self.update_parameter()

    def update_parameter(self):
        # Update impedance of c1
        self.c1_r = 1 / (2 * self.c1 * self.fs)

        # Update parameters of parallel adaptor
        self.p1_r = torch.stack([(self.c1_r * self.vs_r) / (self.c1_r + self.vs_r), self.c1_r, self.vs_r])
        self.p1_g = torch.reciprocal(self.p1_r)
        self.p1 = (2 / torch.sum(self.p1_g)) * torch.ones(3, 1) @ self.p1_g.unsqueeze(0) - torch.eye(3)

        # Update parameter of input impedance of diode pair
        self.dp_r = self.p1_r[0] / 3e3

    def forward(self, v_in, vs_r, fs):
        # waves initiate
        output = torch.zeros_like(v_in)
        p1_a = torch.zeros(3, dtype=torch.float32)
        p1_b = torch.zeros(3, dtype=torch.float32)

        for i in range(len(v_in)):
            self.set_states(vs_r[i], fs[i])

            # Forward Scan
            p1_a[1] = p1_b[1]
            p1_a[2] = v_in[i]
            p1_b[0] = self.p1[0] @ p1_a

            # Local Root Scattering
            dp_a = p1_b[0]
            dp_b = self.dp(torch.stack([dp_a.squeeze(), self.dp_r])).squeeze(0)

            # Backward Scan
            p1_a[0] = dp_b
            p1_b[1] = self.p1[1] @ p1_a
            p1_b[2] = self.p1[2] @ p1_a

            # Read Output
            output[i] = (dp_a + dp_b) / 2

        return output


# %%
def impedance_lookup(fs):
    fs_lookup = {
        tuple([44.1e3]): (1.7e3, 2.5e3),
        tuple([48e3]): (1.7e3, 2.5e3),
        tuple([44.1e3, 48e3]): (1.7e3, 2.5e3),
        tuple([88.2e3]): (0.9e3, 1.3e3),
        tuple([96e3]): (0.9e3, 1.3e3),
        tuple([88.2e3, 96e3]): (0.9e3, 1.3e3),
        tuple([176.4e3]): (0.45e3, 0.65e3),
        tuple([192e3]): (0.45e3, 0.65e3),
        tuple([176.4e3, 192e3]): (0.45e3, 0.65e3),
    }
    return fs_lookup.get(tuple(fs), (0.4e3, 3.0e3))


def get_model_path(path_root="pre_trained_models", circuit_type="dp", num_up: int = 1, num_down: int = 1,
                   diode_type="1N4148", data_type="l", fs=None, num_layers=2, layer_size=16):
    pre_models_path = os.path.join(path_root, f"{num_up}u{num_down}d")
    circuit_type = (
        f"diode_pair_{num_up}u{num_down}d" if circuit_type == "dp" else
        f"diode_clipper_{num_up}u{num_down}d"
    )

    fs = [44.1e3, 48e3] if fs is None else fs
    impedance_begin, impedance_end = impedance_lookup(fs)
    impedance = f"{impedance_begin / 1e3 :.2f}k_{impedance_end / 1e3 :.2f}k"

    network_size = f"{num_layers}x{layer_size}"

    model_name = "_".join([circuit_type, diode_type, data_type, impedance, network_size, "*.pth"])
    model_path = os.path.join(pre_models_path, model_name)
    return glob.glob(model_path)[0]


def train_epoch(model, train_dl, loss_func, skip_size, optimizer, train_mode=True):
    model.train()
    total_loss, total_samples = 0.0, 0
    for x, y in train_dl:
        x = x.squeeze()
        v_in = x[:, 0]
        r_in = x[:, 1]
        fs_in = x[:, 2]

        pred = model(v_in, r_in, fs_in)
        loss = loss_func(pred[skip_size:], y.squeeze()[skip_size:])
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss / total_samples


def eval_model(model, valid_dl, loss_func, skip_size):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for x, y in valid_dl:
            x = x.squeeze()
            v_in = x[:, 0]
            r_in = x[:, 1]
            fs_in = x[:, 2]

            pred = model(v_in, r_in, fs_in)
            loss = loss_func(pred[skip_size:], y.squeeze()[skip_size:])
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    return total_loss / total_samples


def save_model(model, path_root, pre_loss, fine_tuned_loss):
    pre_loss_str = f"{pre_loss:.2e}"
    path_save = dp_path + f"_{loss_str}" + ".pth"
    torch.save(model, path_save)
    print(f"Save model to {path_save}")


def run_model(model, train_dl, valid_dl, epochs, loss_func, skip_size, optimizer, scheduler, train_mode):
    start_time = time.time()
    print("               [Train loss]    [Eval loss]     [Time left]")
    print("----------------------------------------------------------")
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        start_time_epoch = time.time()
        train_loss = train_epoch(model, train_dl, loss_func, skip_size, optimizer, train_mode)
        valid_loss = eval_model(model, valid_dl, loss_func, skip_size)
        scheduler.step(valid_loss)
        end_time_epoch = time.time()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch+1:>5d}\t\t{train_loss:>8.4e}\t\t{valid_loss:>8.4e}\t\t{(end_time_epoch-start_time_epoch)/60*(epochs-epoch-1):>.2f} min")

        # torch.save(model, f"checkpoint_epoch_{epoch}.pth")

    end_time = time.time()
    print(f"\nTime cost：{(end_time - start_time)/60:>.2f} min")

    return train_losses, valid_losses


num_up = 1
num_down = 1

N = 50000
fs = [44.1e3, 48e3]

num_layers = 3
layer_size = 32

r_begin = 10e3
r_end   = 100e3

batch_size = 16
skip_size = 50
real_batch_size = batch_size + skip_size

epochs = 10

dp_path = get_model_path(circuit_type="dp", num_up=num_up, num_down=num_down, data_type="l", fs=fs,
                         num_layers=num_layers, layer_size=layer_size)

model = NDC(dp_path=dp_path).to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
scheduler = ReduceLROnPlateau(optimizer, factor=0.50, verbose=True)
train_dl, valid_dl = get_diodeClipper_dataloader(N, fs, num_up, num_down, r_begin, r_end,
                                                 batch_size=real_batch_size, plot=True)
train_losses, valid_losses = run_model(model, train_dl, valid_dl, epochs, loss_func, skip_size, optimizer, scheduler,
                                       train_mode=True)
