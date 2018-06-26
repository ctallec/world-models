# Pytorch implementation of the "WorldModels"

This page presents a reimplementation of the paper [World Models](https://arxiv.org/pdf/1803.10122.pdf)
(Ha et al., 2018)[1] in pytorch on the [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) environment.
The implementation is available [here](https://github.com/ctallec/world-models).

## Summary of World Models
*World Models* introduces a *model-based* approach to reinforcement learning. It revolves around a three part model, comprised of:

  1. A Variational Auto-Encoder (VAE, Kingma et al., 2014)[2], a generative model, who learns both an encoder and a decoder. The encoder's task is to compress the input images into a compact latent representation. The decoder's task is to recover the original image from the latent representation.
  2. A Mixture-Density Recurrent Network (MDN-RNN, Graves, 2013)[3], trained to predict the latent encoding of the next frame given past latent encodings and actions. The mixture-density network outputs a gaussian mixture observational density at each time step, allowing for multi-modal model predictions.
  3. A simple linear Controller (C). It takes as inputs both the latent encoding of the current frame and the hidden state of the MDN-RNN given past latents and actions and outputs an action. It is trained to maximize the cumulated reward using the Covariance-Matrix Adaptation Evolution-Strategy ([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf), Hansen, 2006)[4].

On a given environment, the model is trained sequentially as follow:
  1. Sample randomly generated rollouts from a well suited *random policy*.
  2. Train the VAE on images drawn from the rollouts.
  3. Train the MDN-RNN on the rollouts. To reduce computational load, we trained the MDN-RNN on fixed size subsequences of the rollouts.
  4. Train the controller while interacting with the environment using CMA-ES.

Alternatively, if the MDN-RNN is good enough at modelling the environment, the controller can be trained directly on simulated rollouts in the dreamt environment.

## Results
On the CarRacing-v0 environment, results were reproducible with relative ease. Our own implementation reached a best score of 860 which is below the 906 reported in the paper, but much better than the second best benchmark reported which is around 780. We believe the gap in the results is related to our reduced computational power, resulting in tamed down hyperparameters for CMA-ES compared to those used in the paper. Gifs displaying the behavior of our best trained model are provided below.
[TODO] Add results.

## Additional experiments
[TODO] Write up.

## Authors

* **Corentin Tallec** - [ctallec](https://github.com/ctallec)
* **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
* **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)


## References
[1] Ha, D. and Schmidhuber, J. World Models, 2018
[2] Kingma, D., Auto-Encoding Variational Bayes, 2014
[3] Graves, A., Generating Sequences With Recurrent Neural Networks, 2013
[4] Hansen, N., The CMA evolution strategy: a comparing review, 2006
