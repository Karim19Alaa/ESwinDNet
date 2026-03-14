# [ESwinDNet: Image Demoiréing Using Multiscale Swin Transformer Layers](https://ieeexplore.ieee.org/document/11084615)

Capturing electronic screens with digital cameras introduces
high-frequency artifacts, known as moiré patterns, degrading overall image quality and colors. This work proposes
ESwinDNet, an image demoiréing model that combines an
encoder-decoder architecture with multiscale Swin Transformer layers. These layers efficiently compute pixel-level
attention, a crucial aspect for low-level vision tasks such as
image demoiréing. The proposed ESwinDNet model achieves
comparable results to the large variant of the baseline model
ESDNet-L on the UHDM dataset, demonstrating its capabilities in the removal of moiré patterns in 4K images, with
nearly half the number of parameters and floating point operations, yielding faster training and inference time

![Moiré pattern removal using ESwinDNet-L](assets/output.gif "Moiré pattern removal using ESwinDNet-L")

## Table Of Content

1. [Installation](#Installation)
2. [Datasets](#Datasets)
3. [Training](#Training)
4. [Testing](#Testing)
5. [Acknowledgement](#Acknowledgement)
6. [Citation](#Citation)

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/Karim19Alaa/ESwinDNet.git
cd ESwinDNet
```

### 2. Create and Activate Virtual Environment

Using **Python 3.10**

```sh
# Create virtual environment
python -m venv eswindnet_env

# Activate it
# For Linux/MacOS:
source eswindnet_env/bin/activate

# For Windows (PowerShell):
eswindnet_env\Scripts\activate
```

### 3. Install Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

Make sure to download UHDM or FHMDi dataset in the `datasets` directory or edit the training scripts to point to their location.

## Training

### Train ESwinDNet on UHDM dataset

```sh
./scripts/train_eswindnet_uhdm.sh
```

```sh
./scripts/train_eswindnetL_uhdm.sh
```

---

### Train ESwinDNet on FHDMi dataset

```sh
./scripts/train_eswindnet_fhdmi.sh
```

```sh
./scripts/train_eswindnetL_fhdmi.sh
```

## Testing

Make sure to edit the scripts to use the desire checkpoint.

### Test ESwinDNet on UHDM dataset

```sh
./scripts/test/test_eswindnet_uhdm.sh
```

```sh
./scripts/test/test_eswindnetL_uhdm.sh
```

---

### Test ESwinDNet on FHDMi dataset

```sh
./scripts/test/test_eswindnet_fhdmi.sh
```

```sh
./scripts/test/test_eswindnetL_fhdmi.sh
```

## Acknowledgement

This repository relies on the work of

- [Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoiréing](https://github.com/CVMI-Lab/UHDM)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [HAT](https://github.com/XPixelGroup/HAT)

And the help of

- [Muhammad Magdy](https://github.com/muhmagdy)
- [Youhanna Yousry](https://github.com/Youhanna-Yousry)

## Citation

```
@INPROCEEDINGS{11084615,
  author={Alaa, Karim and Torki, Marwan},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  title={Eswindnet: Image Demoiréing Using Multiscale Swin Transformer Layers},
  year={2025},
  volume={},
  number={},
  pages={845-850},
  keywords={Wavelet transforms;Training;Image quality;Costs;Image color analysis;Computational modeling;Semantics;Transformers;Digital cameras;Software development management;Image Demoiréing;Swin Transformer;Multiscale Network;Wavelet Transform},
  doi={10.1109/ICIP55913.2025.11084615}}

```
