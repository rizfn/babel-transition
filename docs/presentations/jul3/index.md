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

English speakers can't learn Japanese!

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

![bg fit](../fig/jun10/language_map_india.png)

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

![bg right:32% fit](../fig/may28/agents.png)

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

![bg](../fig/jun10/reverse_babel.png)

$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$

<audio controls preload src="../fig/jun10/v72_40056.mp3"></audio>

---

### Exact evolutionary details:

- Each generation, the $N$ agents play $N_{rounds}$ games.

- Each round, agents are ranomly paired up and play.

- Average fitness is calculated.

- Top $N_{winners}$ of agents reproduce.

- New agents' languages are inherited, but each bit can flip with probability $\mu$.

---

### Evolutionary Parameters

$\\$

$N$ : Number of agents = 1000
$N_{rounds}$ : Number of games per generation = 500
$N_{generations}$ : Number of generations = 1000
$N_{winners}$ : Number of winners (selection pressure) = 500

---

### "Real" Parameters

$\\$

$\alpha$ : Relatedness bonus
$\gamma$ : Discommunication bonus

$B$ : Length of the bit string = 16

$\mu$ : Mutation rate = 0.01



---

![bg fit](../fig/jun10/umap_clusters_g_3_a_1_N_1000_L_16_mu_0.01_gen_1000.png)


---

### Number of communicable people

![width:570px](../fig/jun10/communicable_hist_g_1_a_0_N_1000_L_16_mu_0.01_thresh_1_gen_1000.png)![width:570px](../fig/jun10/communicable_hist_g_1_a_1_N_1000_L_16_mu_0.01_thresh_1_gen_1000.png)

---

![bg fit](../fig/jun10/heatmap_top50_N_1000_L_16_mu_0.01.png)

---

![bg fit right:40%](../fig/jul3/1bit_schematic.png)

Can be explained through a **reward matrix!**

$\\$

For a single bit language, $\vec{c}\in0,1$

---

![bg fit right:60%](../fig/jun10/errorcatastrophe.png)

### Error catastrophe

Increasing $\mu$ changes the system!

(0.01, 0.1,
0.3, 0.5)


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

Local: $\mathcal{F_A} = \alpha \,\mathcal{U}(\vec{c}_A,\vec{c}_B)$

Global: $\mathcal{F_A} =  \gamma \,d_\mathcal{H}(\vec{c}_A,\vec{c}_B)$

$\\$

Each agent plays 4 local games and $L^2$-5 global games

---

### Reproduction

$\\$

Iterate over "winners" with high fitness

For each winner, find a "loser": the weakest among it's four neighbours

Winner "kills and invades" the loser: replace the loser and winner with (mutated) clones of the winner

Newly added agents are frozen until the next timestep

---

<video width="100%" preload controls src="../fig/jul3/L_256_g_1_a_1_B_16_mu_0.001_K_1.mp4"></video>

---

<video width="100%" preload controls src="../fig/jul3/start1_L_256_g_1_a_1_B_16_mu_0.001_K_1.mp4"></video>


---

![bg fit](../fig/jul3/final_lattices_grid_L_256_B_16_mu_0.0001_K_1.png)

---

![bg fit](../fig/jul3/heatmap_sqrtWeightedClusterSize_L_256_B_16_mu_0.001_K_1.png)

---

Doesn't look particularly good...

![width:1000px](../fig/jul3/sqrt_weighted_cluster_size_vs_alpha_g_3.png)


---

![bg fit](../fig/jul3/final_lattices_grid_L_256_B_16_mu_0.001_K_1.png)

---

![bg fit](../fig/jul3/final_lattices_grid_L_256_B_16_mu_0.01_K_1.png)

---

![bg fit](../fig/jul3/final_lattices_grid_L_256_B_16_mu_0.1_K_1.png)

---

![bg fit](../fig/jul3/zoomed_out_raster_lattices_L_256_B_16_gamma_1_K_1.png)

---

![bg fit](../fig/jul3/alpha_raster_lattices_L_256_B_16_gamma_1_K_1.png)


---


### Future Work

$\\$

Find a better solution for clustering!

Quantify error catastrophe

Look for correlation lengths
