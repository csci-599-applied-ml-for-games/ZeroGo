# ZeroGo

## Current Version: Neural Net Agent
* book_model: small network, 100 games, 5 epoches
* model1s: large network, 1000 games, 50 epoches
* model2: large network, 5000 games, 50 epoches

## Python version bug
* h5 files matched with specific python version (Python3.6 created h5 file cannot be read by Python3.7)
* Use save_model.py to save your version of h5 file

## NN_end_to_end.py
* end to end script from downloading data to open web app
* steps:
1. set encoder
2. set data processor/generator
3. construct NN layers
4. compile model and train
5. save model
6. initiate deep learning agent
7. start web app