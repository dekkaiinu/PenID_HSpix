# Hyperspectral Image Dataset for Individual Penguin Identification



This repository contains the experimental code for the paper accepted at IGARSS2024. The purpose of this code is to provide a reference for the methods and results discussed in the paper.

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
python pix_classification/train.py
```
This will start the training process as configured in your `config.yaml` file under the `cfg` directory. 
The weights and training logs will be saved in `runs/<date>/weight.pt`, where `<date>` is the timestamp of the training session.

2. To test the model, execute:
```
python pix_classification/test.py --model_path <model_path>
```
Ensure that the model weights (`weight.pt`) and configuration used for testing match those used during training. 
Specify the path to the weights as needed in the testing script.

## Dataset

This project utilizes datasets uploaded on Hugging Face. The primary dataset used can be found [here](https://huggingface.co/datasets/dekkaiinu/hyper_penguin_pix).

Additionally, this hyperspectral pixel dataset was extracted from the [hyperspectral image dataset](https://huggingface.co/datasets/dekkaiinu/hyper_penguin) using the script located at `dataset/extract_pix_dataset_from_hsi.py`.

## Citation

```
comming soon
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) 4.0 International License](./LICENSE-CC-BY-NC-ND-4.0.md).

Choose this license if you want to permit others to share (mirror) your mod content, providing that they credit you and don't use your work for commercial purposes.

You can view additional details on [this page](https://creativecommons.org/licenses/by-nc-nd/4.0/), which you should link to in your readme.

## Contact

For any questions or issues, please open an issue on this repository or contact the authors directly at [24amj29@ms.dendai.ac.jp](mailto:24amj29@ms.dendai.ac.jp).