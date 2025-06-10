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

![bg right:56% fit](../fig/may28/babel_tower.png)

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

![bg fit](../fig/jun10/language_map_india.png)

---

## Diversity & Complexity

$\\$

- English speakers can't learn Japanese!

- While we can communicate through body language, it's not as easy to do so through spoken language!

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

$\mathcal{U}(\vec{c}_1, \vec{c}_2) = \mathrm{AND}(\vec{c}_1, \vec{c}_2) = 1$


---

![bg right:32% fit](../fig/may28/agents.png)

### Evolutionary Algorithm

$\\$

- Start with agents, each speaks their own language
- Agents compete and gain fitness
- Fittest agents reproduce

---

### From last time

$\\$

$$ \mathcal{F_A} = \underbrace{\lVert \vec{c}_A \rVert_1}_\textrm{complexity} + \underbrace{\gamma \,d_\mathcal{H}(\vec{c}_A,\vec{c}_B)}_\textrm{discommunication} + \underbrace{\alpha \,d_\mathcal{G}(A, B)}_\textrm{family} $$

$d_\mathcal{G}$ doesn't work. 

Depends heavily on $\beta$, selection pressure.

---

### Fitness Function

$\\$

Agents $A$ and $B$ compete against each other.

$\\$

$$ \mathcal{F_A} =  \underbrace{\gamma \,d_\mathcal{H}(\vec{c}_A,\vec{c}_B)}_\textrm{discommunication} + \underbrace{\alpha \,\mathcal{U}(\vec{c}_A,\vec{c}_B))}_\textrm{understandability} $$

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

$N$ : Number of agents
$N_{rounds}$ : Number of games per generation
$N_{generations}$ : Number of generations
$\mu$ : Mutation rate
$N_{winners}$ : Number of winners (selection pressure)

---

### "Real" Parameters

$\\$

$\alpha$ : Relatedness bonus
$\gamma$ : Discommunication bonus

$B$ : Length of the bit string

---

### Analyzing resultant languages

$\\$

Need a nice **clustering algorithm!**

DBSCAN has a problem, if A links with B, and B links with C, then A and C are linked!

Clique based clustering is NP!

---

## Results

$\\$

......still not a lot...

---

![bg fit](../fig/jun10/umap_clusters_g_3_a_1_N_1000_L_16_mu_0.01_gen_1000.png)


---

### Number of communicable people

![width:570px](../fig/jun10/communicable_hist_g_1_a_0_N_1000_L_16_mu_0.01_thresh_1_gen_1000.png)![width:570px](../fig/jun10/communicable_hist_g_1_a_1_N_1000_L_16_mu_0.01_thresh_1_gen_1000.png)

---

![bg fit](../fig/jun10/heatmap_top50_N_1000_L_16_mu_0.01.png)

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

Each agent plays 4 local games and 4$r$ global games, with random opponents

---

### Reproduction

$\\$

Iterate over "losers" with low fitness

For each loser, look for the fittest agent in radius $K$

Winner "kills and invades" the loser: replace the loser and winner with (mutated) clones of the winner

Newly added agents are frozen until the next timestep

---

### $\alpha$ dominant

$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$

$\alpha=1, \;\gamma=1, \;r=0.5$


![bg fit](../fig/jun10/umap_clusters_L_100_g_1_a_1_r_0.5_B_16_mu_0.01_K_3.png)

---

### $\gamma$ dominant

$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$
$\\$

$\alpha=1, \;\gamma=1, \;r=2$

![bg fit](../fig/jun10/umap_clusters_L_100_g_1_a_1_r_2_B_16_mu_0.01_K_3.png)

---

### Heatmaps for $r=1$

![width:1100px](../fig/jun10/heatmap_latticeNbrVsGlobal_gir_1.png)


---

### Heatmaps for $r=2$

![width:1100px](../fig/jun10/heatmap_latticeNbrVsGlobal_gir_2.png)


---

### Heatmaps for $r=0.5$

![width:1100px](../fig/jun10/heatmap_latticeNbrVsGlobal_gir_0.5.png)

---

### Future Work

$\\$

Find a better solution for clustering!

Quantify error catastrophe

Look for correlation lengths
