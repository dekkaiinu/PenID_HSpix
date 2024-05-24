# Hyperspectral Image Dataset for Individual Penguin Identification
  <p align="center">
    <a href='https://arxiv.org/abs/2405.14146'>
      <img src='https://img.shields.io/badge/arXiv-2405.14146-b31b1b.svg' alt='arXiv PDF'>
    </a>
    <a href='https://033labcodes.github.io/igrass24_penguin/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/datasets/dekkaiinu/hyper_penguin_pix'>
      <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue' alt='HuggingFace'>
    </a>
  </p>

This repository is an official implementation of the paper "Hyperspectral Image Dataset for Individual Penguin Identification" accepted to IGARSS2024.

## Installation

This project uses Poetry for dependency management. Install dependencies using Poetry:

```
poetry install
```
This will create a virtual environment and install all the necessary packages as specified in the `pyproject.toml` file, including specific versions of `torch` and `torchvision` for CUDA 11.1 support. 
Ensure that your `torch` environment is compatible with your CUDA version. 
You may need to adjust the `torch` and `torchvision` versions in the `pyproject.toml` file to match your CUDA installation.

## Usage

To run the experiments, execute:
1. To train the model, run the following command from the project root:
```
cd pix_classification
python train.py
```
This will start the training process as configured in your `config.yaml` file under the `cfg` directory. 
The weights and training logs will be saved in `runs/<date>/weight.pt`, where `<date>` is the timestamp of the training session.

2. To test the model, execute:
```
cd pix_classification
python test.py --model_path <model_path>
```
Ensure that the model weights (`weight.pt`) and configuration used for testing match those used during training. 
Specify the path to the weights as needed in the testing script.

## Dataset

This project utilizes datasets uploaded on Hugging Face. The primary dataset used can be found [here](https://huggingface.co/datasets/dekkaiinu/hyper_penguin_pix).

Additionally, this hyperspectral pixel dataset was extracted from the [hyperspectral image dataset](https://huggingface.co/datasets/dekkaiinu/hyper_penguin) using the script located at `dataset/extract_pix_dataset_from_hsi.py`.

## Citation

```
@misc{noboru2024hyperspectral,
      title={Hyperspectral Image Dataset for Individual Penguin Identification}, 
      author={Youta Noboru and Yuko Ozasa and Masayuki Tanaka},
      year={2024},
      eprint={2405.14146},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) 4.0 International License](./LICENSE-CC-BY-NC-ND-4.0.md).

Choose this license if you want to permit others to share (mirror) your mod content, providing that they credit you and don't use your work for commercial purposes.

You can view additional details on [this page](https://creativecommons.org/licenses/by-nc-nd/4.0/), which you should link to in your readme.

## Contact

For any questions or issues, please open an issue on this repository or contact the authors directly at [24amj29@ms.dendai.ac.jp](mailto:24amj29@ms.dendai.ac.jp).