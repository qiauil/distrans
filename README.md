<h1 align="center">
  <img src="./assets/flow.png" width="128"/>
</h1>
<h4 align="center">distrans</h4>
<h6 align="center">A PyTorch Library for Distribution Transformation in Generative Modeling</h6>

## Installation

* Install the latest version through pip: `pip install git+https://github.com/qiauil/distrans`
* Install locally: Download the repository and run `./install.sh` or `pip install .`

## Playground

* Distribution transformation from Gaussian: [gaussian.ipynb](https://github.com/qiauil/distrans/blob/main/gaussian.ipynb)
* Transform moons distribution to swiss distribution: [moons2swiss.ipynb](https://github.com/qiauil/distrans/blob/main/moons2swiss.ipynb)

## References
### Diffusion
* [***Denoising Diffusion Probabilistic Models***, Jonathan Ho, Ajay Jain and Pieter Abbeel, NeurIPS, 2020](https://arxiv.org/abs/2006.11239)
* [***Diffusion Models Beat GANs on Image Synthesis***, Prafulla Dhariwal and Alex Nichol, NeurIPS, 2021](https://arxiv.org/abs/2105.05233)
### Flow Matching
* [***Flow Matching for Generative Modeling***, Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le, ICLR, 2023](https://arxiv.org/abs/2210.02747)
* [***Improving and generalizing flow-based generative models with minibatch optimal transport***, Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf and Yoshua Bengio, TMLR, 2024](https://arxiv.org/abs/2302.00482)
* [***An introduction to Flow Matching***, Fjelde, Tor and Mathieu, Emile and Dutordoir, Vincent, 2024](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)