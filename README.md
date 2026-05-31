# Fourier-KAN-Based Implicit Full-Waveform Inversion

This repository provides an implementation of Fourier-KAN-based implicit full-waveform inversion (IFWI) for seismic velocity reconstruction.

The code is developed based on an implicit neural representation framework for full-waveform inversion. In this version, a Fourier Kolmogorov-Arnold Network (FKAN) layer is introduced into the coordinate-based neural representation to improve the spectral representation ability of the velocity model during waveform inversion.

## Overview

Full-waveform inversion (FWI) reconstructs subsurface velocity models by minimizing the mismatch between observed and simulated seismic data. Conventional FWI can provide high-resolution results, but it is often sensitive to the initial velocity model and may suffer from local minima.

Implicit full-waveform inversion (IFWI) represents the velocity model using a coordinate-based neural network. Instead of directly updating each velocity grid point, the velocity model is parameterized as a continuous neural function of spatial coordinates.

The basic workflow is:

```text
Spatial coordinates → Neural representation → Velocity model → Wave-equation modeling → Predicted seismic data
```

In this repository, the standard implicit neural representation is extended by introducing a Fourier-KAN layer. The FKAN model is trained from random initialization and does not rely on pretrained neural-network weights.

## Main Features

* Fourier-KAN-based implicit neural representation for seismic velocity reconstruction
* Conventional FWI, SIREN-based IFWI, and FKAN-based IFWI examples
* Differentiable acoustic wave-equation modeling
* Coordinate-to-velocity neural parameterization
* Segmented time-domain wavefield modeling
* Total variation regularization for stabilizing inversion
* Synthetic Marmousi and salt-model experiments

## Repository Structure

```text
.
├── ChebyKANLayer.py
├── generator.py
├── ifwi_modules.py
├── rnn_fd.py
├── plot_functions.py
├── IFWI-Marmousi_I-example-salt_fwi.ipynb
├── IFWI-Marmousi_I-example-salt_siren.ipynb
├── IFWI-Marmousi_I-example-salt_kan.ipynb
├── model_init_salt.dat
├── hess_vel.dat
└── figures/
```

## File Description

### `ifwi_modules.py`

This is the core module of the repository. It contains the main implementation of the implicit full-waveform inversion framework.

It includes:

* `IRN`: standard implicit representation network
* `FourierKANLayer`: Fourier-basis KAN feature mapping layer
* `FKANLayer`: Fourier-KAN layer with layer normalization
* `FourierKAN_INR`: FKAN-based implicit neural representation
* `IFWI2D`: main inversion framework supporting `FWI`, `IRN`, and `IFWI` modes

### `rnn_fd.py`

This file implements the differentiable acoustic forward modeling module. It receives the velocity model, source wavelet, source locations, and receiver locations, and outputs predicted shot records.

### `generator.py`

This file provides wavelet and data-segment generation functions. It supports Ricker, Gaussian, and Ormsby wavelets, as well as segmented shot-record generation for truncated time-domain modeling.

### `ChebyKANLayer.py`

This file contains a Chebyshev-KAN layer implementation. It is retained as a KAN-related module and can be used for further comparison or extension.

### `plot_functions.py`

This file provides plotting utilities for displaying velocity models and inversion results.

### Example notebooks

* `IFWI-Marmousi_I-example-salt_fwi.ipynb`: conventional FWI example
* `IFWI-Marmousi_I-example-salt_siren.ipynb`: SIREN-based IFWI example
* `IFWI-Marmousi_I-example-salt_kan.ipynb`: FKAN-based IFWI example

## Method

The FKAN-based IFWI method represents the velocity model as:

```text
v(x, z) = N_theta(x, z)
```

where `(x, z)` denotes spatial coordinates and `N_theta` is the FKAN-based implicit neural representation.

The predicted velocity model is used in the wave-equation modeling operator to generate simulated seismic data. The inversion objective is written as:

```text
min_theta || d_pred(theta) - d_obs ||_2^2 + lambda R(v)
```

where `d_pred` is the predicted seismic data, `d_obs` is the observed seismic data, and `R(v)` denotes the regularization term.

In the current implementation, total variation regularization is used to stabilize the reconstructed velocity model.

## Installation

Create a Python environment:

```bash
conda create -n fkan_ifwi python=3.9
conda activate fkan_ifwi
```

Install the required packages:

```bash
pip install numpy scipy matplotlib pillow jupyter
pip install torch torchvision
pip install deepwave
```

If GPU acceleration is used, please install the PyTorch version that matches your CUDA version.

## Usage

Clone the repository:

```bash
git clone https://github.com/your-username/FKAN-IFWI.git
cd FKAN-IFWI
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run the FKAN-based IFWI example:

```text
IFWI-Marmousi_I-example-salt_kan.ipynb
```

For comparison, you can also run:

```text
IFWI-Marmousi_I-example-salt_fwi.ipynb
IFWI-Marmousi_I-example-salt_siren.ipynb
```

## Example Results

Example velocity reconstruction results can be placed in the `figures/` folder.

```markdown
![Velocity reconstruction result](figures/Vp_true.png)
```

## Notes

* The current version focuses on acoustic full-waveform inversion.
* The FKAN model is optimized directly by waveform misfit.
* No supervised pretraining is required.
* Large checkpoints and intermediate inversion results are recommended to be stored outside the GitHub repository or managed with Git LFS.
* Please modify data paths in the notebooks according to your local environment.

Acknowledgements

This repository is developed based on the original implicit full-waveform inversion framework provided by Jian Sun.

I sincerely thank Jian Sun for providing the original IFWI implementation, including the implicit neural representation structure, wavelet and data generator, segmented modeling strategy, and early example scripts. These components provided an important foundation for the development of the FKAN-based IFWI method in this repository.

The present version extends the original IFWI framework by introducing a Fourier-KAN-based coordinate representation for seismic velocity reconstruction. The FKAN module and related experimental scripts were developed by Wenbin Tian.

The original IFWI framework is closely related to the work of Sun et al. (2023), which proposed implicit seismic full-waveform inversion with deep neural representations.

The differentiable wave-equation modeling in the current version is implemented with Deepwave. The KAN-related implementation is inspired by recent developments in Kolmogorov-Arnold Networks and Fourier/Chebyshev basis representations.

References

Sun, J., Innanen, K. A., Zhang, T., and Trad, D. O., 2023. Implicit seismic full waveform inversion with deep neural representation. Journal of Geophysical Research: Solid Earth, 128(3), e2022JB025964. https://doi.org/10.1029/2022JB025964


## Author

Wenbin Tian
China University of Petroleum (Beijing), Karamay Campus
May 31, 2026

## Citation

If you use this code in your research, please cite the related paper:

```bibtex
@article{tian2026fkanifwi,
  title   = {Adaptive Spectral Parameterization for Seismic Implicit Full-Waveform Inversion Using Fourier Kolmogorov-Arnold Networks},
  author  = {Tian, Wenbin},
  journal = {To be updated},
  year    = {2026}
}
```

## License

This repository is released for academic research purposes. Please add a specific open-source license, such as MIT, Apache-2.0, or GPL-3.0, before public release.
