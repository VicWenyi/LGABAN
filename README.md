# LGABAN: Integrating Multi-Scale Graph and Sequence Features with Attention Networks for enhanced prediction of drug-protein interactions

## Framework
<img width="2184" height="1309" alt="LGABAN流程图" src="https://github.com/user-attachments/assets/042f7fa0-bb35-432c-90d8-245dd9b1afa0" />


## System Requirements
The source code developed in Python 3.8 using PyTorch 2.4.0. The required python dependencies are given below. LGABAN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```
torch>=2.4.0
dgl>=2.4.0
dgllife>=0.2.8
numpy>=1.24.1
scikit-learn>=1.3.2
pandas>=2.0.3
prettytable>=3.11.0
rdkit~=2021.03.2
yacs~=0.1.8
```
## Installation Guide
Clone this Github repo and set up a new conda environment.
```
# create a new conda environment
$ conda create --name LGABAN python=3.8
$ conda activate LGABAN

# install requried python dependencies
$ conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch
$ conda install -c dglteam/label/th24_cu118 dgl
$ conda install -c conda-forge rdkit==2021.03.2
$ pip install dgllife==0.2.8
$ pip install -U scikit-learn
$ pip install yacs
$ pip install prettytable

# clone the source code of LGABAN
$ git clone https://github.com/VicWenyi/LGABAN.git
$ cd LGABAN
```


## Datasets
The `datasets` folder contains all experimental data used in LGABAN: [BindingDB](https://www.bindingdb.org/bind/index.jsp) , [BioSNAP](https://github.com/kexinhuang12345/MolTrans) , [Human](https://github.com/lifanchen-simm/transformerCPI) , [Davis](https://github.com/LBBSoft/DeepCDA.git). 


## Run LGABAN on Our Data to Reproduce Results

To train LGABAN, where we provide the basic configurations for all hyperparameters in `config.py`. 

For the in-domain experiments with vanilla LGABAN, you can directly run the following command. `${dataset}` could either be `bindingdb`, `biosnap`, `human` and `Davis`. `${split_task}` could be `random` and `cold`. 
```
$ python main.py --cfg "configs/LGABAN.yaml" --data ${dataset} --split ${split_task}
```
