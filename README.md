# Pytorch implementation of the "WorldModels"


Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631


## Prerequisites

The implementation is based on Python3 and PyTorch, check their website [here](https://pytorch.org) for installation instructions. The rest of the requirements is included in the [requirements file](requirements.txt), to install them:
```bash
pip3 install -r requirements.txt
```

## Running the worldmodels

The model is composed of three parts:

  1. The Variational Auto-Encoder (VAE), which task is to compress the input images into a compact latent representation.
  2. The Mixture-Density Network mixed with a Recurrent Neural Network (MDN-RNN), trained to predict the (compressed) frames over time.
  3. The Controller (C), built upon the VAE and the MDN-RNN and maximizing the reward given the output of other modules. It is trained using the Covariance-Matrix Adaptation Evolution-Strategy ([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)) from the `cma` python package.

In the given code, all three sections are trained separately, using the scripts `trainvae.py`, `trainmdrnn.py` and `traincontroller.py`.

The scripts take as argument:
* **--logdir** : The directory in which the models will be stored. If the logdir specified already exists, it loads the old model and continues the training.
* **--noreload** : If you want to override the *logdir* specified with a new model, add this option.

### 1. Data generation
The data is generated using a continuous policy, instead of the random `action_space.sample()` policy from gym, providing more consistent rollouts.

The generation script is `data/carracing.py` and its multiprocessed version `data/generation_script.py`. An example of commands would be:

```bash
python data/carracing.py --rollouts 1000 --dir exp_dir
# or with multiprocessing:
python data/generation_script.py --rollouts 1000 --dir exp_dir --threads 8
```

### 2. Training the VAE
The VAE is trained using the `trainvae.py` file:
```bash
python trainvae.py --logdir exp_dir
```
### 3. Training the MDN-RNN
The MDN-RNN is trained using the `trainmdrnn.py` file, given that the VAE has been trained:
```bash
python trainmdrnn.py --logdir exp_dir
```
### 4. Training and testing the Controller
Finally, we train the controller using CMA-ES:
```bash
python traincontroller.py --logdir exp_dir
```
and the test is made using the `test_controller.py` file:
```bash
python test_controller.py --logdir exp_dir
```



## Authors

* **Corentin Tallec** - [ctallec](https://github.com/ctallec)
* **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
* **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
