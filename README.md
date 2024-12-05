
# New World

![D6148F62-750D-4C4D-B2DA-FF8A88567F63_1_105_c](https://github.com/user-attachments/assets/449cad55-f5c5-4453-87a9-d48fa17b643c)

### ANDHRA Bandersnatch: Training Neural Networks to Predict Parallel Realities
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/andhra-bandersnatch-training-neural-networks/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=andhra-bandersnatch-training-neural-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/andhra-bandersnatch-training-neural-networks/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=andhra-bandersnatch-training-neural-networks)

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- wandb, version 0.12.21


## Training: 


python mainBS1GR3_10_1.py

python mainBS1GR3_100_1.py


python mainAB2GR3_10_1.py

python mainAB2GR3_100_1.py


## Testing

python test_10_BS.py

python test_100_BS.py

python test_multi_10.py

python test_multi_100.py


# Reproducing the results

(1) Train each network five times by changing the checkpoint name -- (Run1, Run2,...). 

(2) After training,  use the test scripts for evaluating the networks. Use appropriate test scripts according to the total heads in the network. For a baseline, the testing script is denoted with "_BS" (test_10_BS.py & test_100_BS.py). The ensemble predictions are denoted with _multi (test_multi_10.py & test_multi_100.py). Make sure the correct checkpoint path is set for the network.

(3) Feed the results from five runs into the t-test.py script, which performs a statistical significance test with mean & standard deviation calculation. 

### If you encounter any issues reproducing the main paper's results, please don't hesitate to open an issue. 

### Code reference

https://github.com/kuangliu/pytorch-cifar




