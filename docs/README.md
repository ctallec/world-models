# Pytorch implementation of the "WorldModels"

This page presents a reimplementation of the paper [World Models](https://arxiv.org/pdf/1803.10122.pdf)
(Ha et al., 2018)[1] in pytorch on the [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) environment.
The implementation is available [here](https://github.com/ctallec/world-models).


## Summary of World Models

*World Models* introduces a *model-based* approach to reinforcement learning. It revolves around a three part model, comprised of:

  1. A Variational Auto-Encoder (VAE, Kingma et al., 2014)[2], a generative model, who learns both an encoder and a decoder. The encoder's task is to compress the input images into a compact latent representation. The decoder's task is to recover the original image from the latent representation.
  2. A Mixture-Density Recurrent Network (MDN-RNN, Graves, 2013)[3], trained to predict the latent encoding of the next frame given past latent encodings and actions. The mixture-density network outputs a gaussian mixture observational density at each time step, allowing for multi-modal model predictions.
  3. A simple linear Controller (C). It takes as inputs both the latent encoding of the current frame and the hidden state of the MDN-RNN given past latents and actions and outputs an action. It is trained to maximize the cumulated reward using the Covariance-Matrix Adaptation Evolution-Strategy ([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf), Hansen, 2006)[4].

![Architecture]({{ site.url }}/img/arch_fig.png)

On a given environment, the model is trained sequentially as follow:
  1. Sample randomly generated rollouts from a well suited *random policy*.
  2. Train the VAE on images drawn from the rollouts.
  3. Train the MDN-RNN on the rollouts. To reduce computational load, we trained the MDN-RNN on fixed size subsequences of the rollouts.
  4. Train the controller while interacting with the environment using CMA-ES.

Alternatively, if the MDN-RNN is good enough at modelling the environment, the controller can be trained directly on simulated rollouts in the dreamt environment.


## Reproductibility on the CarRacing environment

On the CarRacing-v0 environment, results were reproducible with relative ease. We were pleasantly surprised to observe that the model achieved good results the first time, since it is well-known that many reinforcement learning results are hard to reproduce, and not stable [5]. Our own implementation reached a best score of 860 which is below the 906 reported in the paper, but much better than the second best benchmark reported which is around 780. We believe the gap in the results is related to our reduced computational power, resulting in tamed down hyperparameters for CMA-ES compared to those used in the paper. Gifs displaying the behavior of our best trained model are provided below.
[TODO] Add results.


![Full model with trained MDRNN]({{ site.url }}/img/trained.gif)

## Additional experiments

We wanted to test the impact of the MDRNN on the results. Indeed, we observed during training that the final loss of the recurrent network was high :

| Method | Loss of the MDRNN |
|--------|---------------|
| Initialization | TODO |
| After training | TODO |

In the original paper, the authors compare their results with a model without the MDRNN, and obtain the following scores :

| Method | Average score |
|--------|---------------|
| Full World Models | 906 ± 21 |
| without MDRNN | 632 ± 251 |

We did an additional experiment. We tested the full world model architecture, but without training the MDRNN, and keeping its random initial weights. We obtained the following results :

| Method | Average score |
|--------|---------------|
| With a trained MDRNN | TODO |
| With an untrained MDRNN | TODO |

![Full model with untrained MDRNN]({{ site.url }}/img/untrained.gif)

It seems that the training of the MDRNN does not improve the performance. Our interpretation of this phenomena is that even if the recurrent model is not able to predict the next state of the environment, its outputs contains some necessary informations for the problem. The first-order informations such as the velocity of the car are not contained in a single frame. Therefore, a strategy learned without the MDRNN cannot use it. But it seems reasonable that even a random MDRNN still keeps some information on the velocity, and that it is enough for learning a good strategy on this problem.


## Conclusion

TODO


## Authors

* **Corentin Tallec** - [ctallec](https://github.com/ctallec)
* **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
* **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## References

[1] Ha, D. and Schmidhuber, J. World Models, 2018

[2] Kingma, D., Auto-Encoding Variational Bayes, 2014

[3] Graves, A., Generating Sequences With Recurrent Neural Networks, 2013

[4] Hansen, N., The CMA evolution strategy: a comparing review, 2006

[5] Irpan, A., [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html), 2018
