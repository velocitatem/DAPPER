# LSNet: See Large, Focus Small

This is an implementation of the LSNet architecture as described in the paper [LSNet: See Large, Focus Small](https://arxiv.org/abs/2503.23135).

## Overview

LSNet is a new family of lightweight vision models inspired by the dynamic heteroscale capability of the human visual system, i.e., "See Large, Focus Small". The architecture combines large-kernel perception and small-kernel aggregation to efficiently capture a wide range of perceptual information and achieve precise feature aggregation.

## Architecture

The LSNet architecture consists of the following components:

1. **LS Convolution**: A novel convolution operation that combines large-kernel perception and small-kernel aggregation.
2. **LS Block**: A building block that consists of LS Convolution followed by a residual connection.
3. **LSNet**: The main model that uses LS Blocks to achieve efficient and effective feature extraction and classification.

## Models

We provide three variants of the LSNet model:

- **LSNet-T (Tiny)**: A lightweight model with 11.4M parameters and 0.3G FLOPs.
- **LSNet-S (Small)**: A medium-sized model with 16.1M parameters and 0.5G FLOPs.
- **LSNet-B (Base)**: A larger model with 23.2M parameters and 1.3G FLOPs.

## Usage

### Downloading Pre-trained Models

You can download pre-trained LSNet models from Hugging Face using the `download_lsnet.py` script:

```bash
python classification/download_lsnet.py --model lsnet_t --output_dir models
```

Available models:
- `lsnet_t`: LSNet-Tiny
- `lsnet_s`: LSNet-Small
- `lsnet_b`: LSNet-Base

Add the `--distilled` flag to download the distilled version of the model.

### Testing the Model

You can test the LSNet model on a sample image using the `test_lsnet.py` script:

```bash
python classification/test_lsnet.py --model lsnet_t --image path/to/image.jpg --show_image
```

### Training the Model

To train the LSNet model, use the `main.py` script with the LSNet configuration:

```bash
python classification/main.py --config classification/config/lsnet.yaml
```

## Configuration

The LSNet configuration is defined in `classification/config/lsnet.yaml`. You can modify this file to change the model size, training parameters, etc.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@misc{wang2025lsnetlargefocussmall,
      title={LSNet: See Large, Focus Small}, 
      author={Ao Wang and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
      year={2025},
      eprint={2503.23135},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23135}, 
}
```

## License

This implementation is released under the same license as the original paper. 