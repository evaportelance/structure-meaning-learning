# Reframing linguistic bootstrapping as joint inference using visually-grounded grammar induction models
 
This repository contains all the code and preprocessed data for models and evaluations presented in the paper [*Reframing linguistic bootstrapping as joint inference using visually-grounded grammar induction models*](./writeups/bootstrapping_manuscript_clean.pdf) by Eva Portelance, Siva Reddy, and Timothy J. O'Donnell. 2024.

## Data

All the necessary preprocessed Abstract Scenes data is available in the [preprocessed-data/abstractscenes directory](./preprocessed-data/abstractscenes/). 

The original Abstract Scenes dataset is publicly available at [http://optimus.cc.gatech.edu/clipart/](http://optimus.cc.gatech.edu/clipart/) .

## Model 

The visually-grounded grammar induction model code is available in the folder [./vc-pcfg](./vc-pcfg/). This folder contains a custom modified version of the VC-PCFG model by [Zhao & Titov (2020)](https://aclanthology.org/2020.emnlp-main.354/).

## Evaluation

All the evaluation and result analysis scripts and results from experiment reported in the paper are available in the folder [./analyses/](./analyses/).

## To run code yourself
Use the following to set up a compatible environment. 

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Anaconda package manager (ideally)
- GPU (for faster training, optional but recommended)

### Installation

Clone the repository:

```bash
git clone https://github.com/evaportelance/structure-meaning-learning.git
```
If you use anaconda, you can clone our environment using the conda-env.txt file:
```bash
cd structure-meaning-learning
conda create --name myvcpcfgenv --file ./conda-env-graminduct.txt
conda activate myvcpcfgenv
pip install requirements.txt

```

The grammar induction model training requires a custom version of Torch-Struct:
```bash
git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git
cd pytorch-struct
pip install -e .
```

### Training the self-supervised image encoder

Note that the pretrained image embeddings are already available in the preprocessed-data folder. If you would like to retrain your own version of the self supervised image encoder, run the following code. This code expects the original Abstract scenes to have the following relative path '../../AbstractScenes_v1.1/RenderedScenes/', but this can by changed in the configs.py file.

```bash
cd structure-meaning-learning/pytorch-simclr
conda activate myvcpcfgenv
python ./simclr.py --dataset 'abstractscenes' --batch-size 100 --num-epochs 500 --cosine-anneal --test-freq 5

```

### Training visually grounded grammar induction models
Use the following commands to train a model.

#### Joint-learning model with self-supervised image embeddings

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int]
```

#### Joint-learning model with visual-labels as embeddings

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int]
```

#### Semantics-first model

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int] --sem_first
```

#### Syntax-first model

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int] --syn_first
```

To train any of these models on the out-of-distribution split where all instances of test verbs are held out, simply add the `--one_shot' flag to call. 



## Citation

Please cite the following paper:
```
@article{portelance2024reframing,
  title={Reframing linguistic bootstrapping as joint inference using
visually-grounded grammar induction models},
  author={Portelance, Eva and Frank, Michael C. and Jurafsky, Dan},
  year={2024},
  journal={ArXiv preprint: }
}
```

Please also cite the VC-PCFG paper and the Abstract Dataset papers:

```
@inproceedings{zhao-titov-2020-visually,
    title = "Visually Grounded Compound {PCFG}s",
    author = "Zhao, Yanpeng  and
      Titov, Ivan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.354",
    pages = "4369--4379"
}
```
```
@INPROCEEDINGS{zitnick2013learning,
  author={Zitnick, C. Lawrence and Parikh, Devi and Vanderwende, Lucy},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision}, 
  title={Learning the Visual Interpretation of Sentences}, 
  year={2013},
  pages={1681-1688},
  url={https://ieeexplore.ieee.org/document/6751319}}

@INPROCEEDINGS{zitnick2013bringing,
  author={Zitnick, C. Lawrence and Parikh, Devi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}, 
  title={Bringing Semantics into Focus Using Visual Abstraction}, 
  year={2013},
  pages={3009-3016},
  url={https://ieeexplore.ieee.org/document/6619231}}
```

If you use the pytorch-simclr code to retrain the image encoder, please cite the repo it is forked from :

```
@article{
  silva2020exploringsimclr,
  title={Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations},
  author={Silva, Thalles Santos},
  journal={https://sthalles.github.io},
  year={2020}
  url={https://sthalles.github.io/simple-self-supervised-learning/}
}
```
