---
marp: true
theme: uncover
math: mathjax
paginate: true
_paginate: skip
backgroundColor:	#FCEFCB
color: #4E1F00
style: |
        .columns {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.6rem;
        }
        h1, h2, h3, h4, h5, h6, strong {
          color: #8A2D3B;
        }
        section::after {
          text-shadow: 0 0 1px #74512D;
        }

---

![bg right:56%](../fig/may28/babel_tower.png)

# Self-Organized Babel

$\\$

Riz Fernando Noronha

---

## Origin of Language

$\\$

> *Language has evolved as a <br> **means of communication***

$\\$

Is that  really true?

---

![bg](../fig/may28/language_tree.png)

---

![bg](../)
![bg vertical fit](../fig/jul3/different_langauges.png)

### Diversity in Languages

$\\$

Why isn't there a common language?

$\\$

$\\$

$\\$

$\\$

---

### Complexity of Language

$\\$

![bg fit right:35%](../fig/jul3/gender_of_inanimate_objects.png)

eg: Gender of inanimate objects

Is the benefit worth the cost?

---

### Hypothesis:


Language has evolved as a form of **discommunication**

Keep ideas secret from other tribes!

![](../fig/jul3/communicate_only_with_friends.png)


---

## The Model

$\\$

Language can represented as a **bit string** of length $B$
Each bit represents a *feature* or *word* in the language

$\\$

$$\vec{c} = \underbrace{\left[0,1,0,0,0,1,0 \right]}_B$$

---

### Understandability $\mathcal{U}$

$\\$

$$\vec{c}_1 = \left[0,1,0,0,0,1,0 \right]$$

$$\vec{c}_2 = \left[1,1,0,1,0,0,0 \right]$$

- Both languages have a word $\implies$ Understandable
- Only one has a word $\implies$ Not understandable
- Neither have a word $\implies$ Not understandable

$\mathcal{U}(\vec{c}_1, \vec{c}_2) = \sum\mathrm{AND}(\vec{c}_1, \vec{c}_2) = 1$

---

### Discommunication $d_H$

$\\$

$$\vec{c}_1 = \left[0,1,0,0,0,1,0 \right]$$

$$\vec{c}_2 = \left[1,1,0,1,0,0,0 \right]$$


Agents want to be *different* from others.

$d_H(\vec{c}_1, \vec{c}_2) = \sum\mathrm{XOR}(\vec{c}_1, \vec{c}_2) = 3$


---

![bg right:33% fit](../fig/jul28/agents-languages.png)

### Evolutionary Algorithm

$\\$

- Start with agents, each speaks their own language

- Agents compete and gain fitness

- Fittest agents reproduce

---

### Fitness Function

$\\$

Agents $A$ and $B$ compete against each other.

$\\$

$$ \mathcal{F_A} =  \underbrace{\gamma \,\frac{d_\mathcal{H}(\vec{c}_1,\vec{c}_2)}{B}}_\textrm{discommunication} + \underbrace{\alpha \,\frac{\mathcal{U}(\vec{c}_1,\vec{c}_2)}{B}}_\textrm{understandability} $$

---

### Exact evolutionary details:

- Each generation, each agent competes with every other agent, and gains fitness

- Total fitness is calculated.

- Top 50% of agents reproduce.

- New agents' languages are inherited, but each bit can flip with probability $\mu$.

---

### Evolutionary Parameters

$\\$

$N$ : Number of agents = 1000
$N_{generations}$ : Number of generations = 1000
$N_{winners}$ : Number of winners (selection pressure) = 500

---

### "Real" Parameters

$\\$

$\alpha$ : Relatedness bonus

$\gamma$ : Discommunication bonus

$\\$

$B$ : Length of the bit string = 16

$\mu$ : Mutation rate = 0.01


---

![bg fit](../fig/jun10/heatmap_top50_N_1000_L_16_mu_0.01.png)

---

![bg fit right:40%](../fig/jul3/1bit_schematic.png)

Can be explained through a **reward matrix!**

$\\$

For a single bit language, $\vec{c}\in0,1$

---

## Lattice Model

$\\$

The same dynamics, but on a lattice!

Use a square lattice, of $L\times L$

One agent on each lattice site

Agents again play games, and gain fitness

---

![bg fit right:40%](../fig/jun10/lattice_algorithm.png)

Agents have both **local** and **global** interactions.

Local: $\mathcal{F_A} = (\alpha/4) \,\mathcal{U}(\vec{c}_A,\vec{c}_B)$

Global: $\mathcal{F_A} =  (\gamma/N) \,d_\mathcal{H}(\vec{c}_A,\vec{c}_B)$

$\\$

Each agent plays 4 local games and $L^2$ global games

---

![bg fit right:25%](../fig/jul28/reproduction_schematic.png)

### Reproduction


Choose a random site to reproduce

Find the weakest neighbour

Winner "kills and invades" the loser: replace the loser with (mutated) clone of the winner

Newly added agents are frozen until the next timestep

Mutate the entire lattice

---

<iframe width="100%" height="100%" src="https://rizfn.github.io/babel-transition/visualizations/understandabilityVsHamming2D/" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

![bg fit](../fig/jul3/final_lattices_grid_L_256_B_16_mu_0.0001_K_1.png)

---

![bg fit](../fig/jul28/population_timeseries_L_256_B_16_g_1_a_0.8_mu_0.0001.png)


---

![bg fit](../fig/jul3/heatmap_sqrtWeightedClusterSize_L_256_B_16_mu_0.001_K_1.png)

---

![width:1000px](../fig/jul3/sqrt_weighted_cluster_size_vs_alpha_g_3.png)



---

![bg fit](../fig/jul28/alpha_mu_raster_lattices_L_256_B_32_gamma_1.png)

---

![bg fit](../fig/jul28/ones_heatmap_top50_N_1000_B_16_gamma_1.png)

---

![bg fit](../fig/jul28/ones_distribution_grid_top50_N_1000_B_16_gamma_1.png)

---

![bg fit](../fig/jul28/ones_heatmap_L_256_B_16_gamma_1.png)

---

![bg fit](../fig/jul28/ones_distribution_grid_L_256_B_16_gamma_1.png)


