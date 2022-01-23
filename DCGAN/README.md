# Deep Convolutional GAN

Open in Colab!<br>
[![Open DCGAN in
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YooPaul/GANs/blob/master/DCGAN/DCGAN.ipynb)<br>

# Training Objective

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\max_{D}E_{x}[log(D(x))]%20+%20E_{z}[log(1%20-%20D(G(z))]"
/>

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\min_{G}E_{z}[log(1%20-%20D(G(z))]"
/>

## References

[1] Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

[2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
