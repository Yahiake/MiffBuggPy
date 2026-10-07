# MiffBuggPy - Neural Big Muff Pi

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![JUCE](https://img.shields.io/badge/JUCE-8.0-orange.svg)](https://juce.com/)
[![RTNeural](https://img.shields.io/badge/Inference-RTNeural-green.svg)](https://github.com/jatinchowdhury18/RTNeural)
[![PyTorch](https://img.shields.io/badge/Training-PyTorch-red.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**MiffBuggPy** is a real-time, zero-latency neural network emulation of the legendary **Electro-Harmonix Big Muff Pi** fuzz pedal, packaged as a VST3 audio plugin and standalone application for Windows, and Linux.

Trained on high-precision **LiveSPICE** analog circuit simulations (oversampled 16x, 32 Newton-Raphson iterations) using a specialized recurrent neural network architecture, MiffBuggPy delivers **99.63% measured circuit accuracy** with less than **1% CPU usage** per instance.

---

## Key Highlights

- **Studio Reference Fidelity**: Evaluated on completely held-out test audio across thousands of knob positions, achieving an overall **ESR of 0.00371 (< 0.4% error)** and median clip error of **0.24%**.
- **Real-Time Zero Latency**: Recurrent sample-by-sample inference with **0 samples latency** — plug in your guitar and track live with zero delay.
- **Ultra-Lightweight (~390 KB)**: Powered by [RTNeural](https://github.com/jatinchowdhury18/RTNeural) with AVX2 & NEON SIMD acceleration. Consumes less than 1% of a modern CPU core.
- **Physically Decoupled Volume Pot**: In the real Big Muff circuit, Volume is a passive voltage divider directly on the output jack. By mathematically separating Volume from the neural feature vector, 100% of the network's capacity is dedicated to the complex non-linear physics of the clipping stages and tone stack.
- **Hybrid Spectral Loss**: Trained with a combination of Error-to-Signal Ratio (ESR) and Multi-Resolution Short-Time Fourier Transform (MRSTFT) with high-frequency pre-emphasis, ensuring accurate reproduction of high-end fuzz sizzle and pick attack dynamics.

---

## Controls

| Knob | Range | Description |
| :--- | :--- | :--- |
| **Volume** | `0% – 100%` | Output master level. Renormalized so 50% sits at analog unity gain with +6 dB of boost at maximum. Smoothly interpolated per-sample with zero zipper noise. |
| **Sustain** | `0% – 100%` | Controls input drive into the twin diode clipping stages, sweeping from subtle edge-of-breakup grit to massive, squashed, singing fuzz saturation. |
| **Tone** | `0% – 100%` | Blends the passive low-pass and high-pass filter network, sweeping from dark, bass-heavy rumble to the iconic mid-scooped treble bite. |

---

## Installation

Pre-built binaries for Windows, and Linux are automatically compiled on GitHub Releases:

### Windows (x64)
- **VST3**: Copy `MiffBuggPy.vst3` into `C:\Program Files\Common Files\VST3\`
- **Standalone**: Double click `MiffBuggPy.exe` to run directly

### Linux (x86_64)
- **VST3**: Copy `MiffBuggPy.vst3` into `~/.vst3/`
- **Standalone**: Run `MiffBuggPy` directly

---

## Repository Structure

```text
miffbuggpy/
├── CMakeLists.txt                 # CMake build configuration for VST3 and Standalone
├── model.json                     # Pre-trained flagship model embedded into binary
├── src/                           # C++ JUCE & RTNeural plugin source code
│   ├── Plugin.h / Plugin.cpp      # AudioProcessor & real-time inference loop
│   └── Editor.h / Editor.cpp      # Custom GUI editor with responsive knobs
├── training/                      # PyTorch model training and dataset pipeline
│   ├── requirements.txt           # Python dependencies
│   ├── train.py                   # TBPTT trainer, cosine LR schedule, export tool
│   ├── model.py                   # KnobGRU architecture & RTNeural JSON exporter
│   ├── losses.py                  # Hybrid ESR + MRSTFT loss with pre-emphasis
│   ├── knobdata.py                # LiveSPICE database reader & memmapped packer
│   └── test_harness.py            # Automated unit test suite
├── circuit/                       # LiveSPICE circuit schematics & sampling specs
│   ├── Big Muff Pi.schx           # Original circuit schematic
│   └── bigmuff-v3.spec.json       # Stratified sampling specification
├── modules/                       # Vendored C++ submodules (JUCE, RTNeural)
└── dist/                          # Packaged release artefacts (.vst3 and .exe)
```

---

## Building from Source

### Prerequisites
- Windows 10/11 x64
- Visual Studio 2022 / 2026 (MSVC with C++17 support)
- CMake 3.20 or newer

### Build Steps

```powershell
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/Yahiake/MiffBuggPy.git
cd MiffBuggPy

# Configure CMake with Release config and AVX2 enabled
cmake -B build -S . -G "Visual Studio 17 2022" -A x64

# Compile the VST3 and Standalone executable
cmake --build build --config Release --parallel

# Packaged binaries will be in dist/
dir dist
```

---

## Training from Scratch

To reproduce or customize the neural training:

### 1. Install Python Dependencies
```bash
cd training
pip install -r requirements.txt
```

### 2. Verify Unit Tests
```bash
python test_harness.py
```

### 3. Pack the Dataset
Pack raw LiveSPICE renders into a high-speed memory-mapped float32 tensor:
```bash
python train.py prepare --db path/to/livespice/data --pack ./pack --seed 7
```

### 4. Train the Model
```bash
python train.py train --db path/to/livespice/data --pack ./pack --out runs/my_muff --epochs 35 --tbptt 2048
```

### 5. Export to RTNeural JSON
```bash
python train.py export --db path/to/livespice/data --pack ./pack --ckpt runs/my_muff/best.pt --out runs/my_muff
```

Replace `model.json` at the root of the repository with your newly exported model and re-compile the C++ plugin.

---

## Technical Details

### Architecture: `KnobGRU`
- **Layers**:
  - `Dense`: 3 inputs $\rightarrow$ 128 hidden (with `tanh` activation)
  - `GRU`: 128 hidden $\rightarrow$ 128 hidden (1 layer)
  - `Dense`: 128 hidden $\rightarrow$ 1 output (linear)
- **Parameters**: 99,713 float32 weights (~390 KB)
- **Conditioning**: Audio sample, Sustain, Tone (Volume dynamically applied as post-gain)

### Performance Benchmark
- **Validation ESR**: `0.00505` (0.50% error)
- **Held-Out Test ESR**: `0.00371` (0.37% error)
- **Peak Memory**: Fits completely inside CPU L2 cache (~390 KB)

---

## Author & Credits

- **Creator & Developer**: **Yahia Kemari** ([yahiakemari@gmail.com](mailto:yahiakemari@gmail.com))
- **LiveSPICE**: [Dillon Sharlet](https://github.com/dsharlet/LiveSPICE)
- **RTNeural**: [Jatin Chowdhury](https://github.com/jatinchowdhury18/RTNeural)
- **JUCE**: [Raw Material Software / PACE Anti-Piracy](https://juce.com/)

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
