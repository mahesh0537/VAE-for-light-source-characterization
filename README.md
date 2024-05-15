# VAE-for-light-source-characterization

#### This is official repository for the paper "Deep learning-based variational autoencoder for classification of quantum and classical states of light" by Mahesh Bhupati, Abhishek Mall, Anshuman Kumar, and Pankaj K. Jha [Paper](https://arxiv.org/abs/2405.05243)

## Setup
1. Clone the repository
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
3. Run the following command to train the model:
```bash
cd src
./train.sh # To train the model for SPAC & SPAT dataset
./train_spac_spat_coherent_thermal.sh # To train the model for mixSPAC, mixSPAT coherent and thermal states 
./train_moe_spac_spat_coherent_thermal.sh # To train the model for mixSPAC, mixSPAT coherent and thermal states using Mixture of Experts an upgraded model which performes better in some cases (not included in the paper)
```
4. Run the following command to generate tentative plots of the paper:
```bash
cd src
python fig_4.py # To generate the plot for Fig. 4
python fig_5.py # To generate the plot for Fig. 5
python fig_6.py # To generate the plot for Fig. 6
python fig_7.py # To generate the plot for Fig. 7
```

## Citation
If you find this code useful in your research, please consider citing the paper:
```
@misc{bhupati2024deep,
      title={Deep learning-based variational autoencoder for classification of quantum and classical states of light}, 
      author={Mahesh Bhupati and Abhishek Mall and Anshuman Kumar and Pankaj K. Jha},
      year={2024},
      eprint={2405.05243},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```