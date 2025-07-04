## The models

The simulations are evolutionary algorithms. Typically, we have $N$ agents who each speak a language, represented by a bitstring $\vec{c}.$ who are paired up, and compete with each other. Each timestep, they play games with several others and gain fitness. The functional form of the fitness function is usually changed between models, but there also are some 2D models.

A common theme is that agents are penalized for speaking the same language, with a term depending on the hamming distance between the languages, $d_\mathcal{H}\left(\vec{c}_A, \vec{c}_B\right)$

There are a ton of models here. Most failed. A few were interesting, and we focused on them. Namely,

[Understandability Vs Hamming](./understandabilityVsHamming/)

[Understandability Vs Hamming 2D](./understandabilityVsHamming2D/)

Both follow

$$ F = \alpha \left( \vec{c}_A \cdot \vec{c}_B \right) + \gamma \cdot d_\mathcal{H}\left(\vec{c}_A, \vec{c}_B\right)$$
