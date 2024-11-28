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

(1) Train each network for five times by changing the checkpoint name -- (Run1, Run2,...). 

(2) After, training use the test scripts for evlauating the networks, use appropriate test script according to total heads in the network. For baseline testing script is denoted with "_BS" (test_10_BS.py & test_100_BS.py). The ensemble preditions are only implemented for tetsing script with 8 heads (test_multi_10.py & test_multi_100.py. make sure the correct chekpoint path is set for the network.

(3) Feed the results from five runs into the t-test.py script, that performs a statistical significance test with mean & standard deviation calculation. 


### Code reference

https://github.com/kuangliu/pytorch-cifar
