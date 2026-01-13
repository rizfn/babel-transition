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

- Start with agents on a lattice of $L\times L$, each speaking their own language

- Agents compete and gain fitness

- Fittest agents reproduce

---

![bg fit right:40%](../fig/jun10/lattice_algorithm.png)

Agents have both **local** and **global** interactions.

Local: $\mathcal{F_A} = (\alpha/4) \,\mathcal{U}(\vec{c}_A,\vec{c}_B)$

Global: $\mathcal{F_A} =  (\gamma/N) \,d_\mathcal{H}(\vec{c}_A,\vec{c}_B)$

$\\$

Each agent plays 4 local games and $L^2$ global games


---

### Fitness Function

$\\$

$$ 
\mathcal{F_A} = \underbrace{\frac\alpha4 \,\sum_{x\in\mathrm{nbrs}}\mathcal{U}(\vec{c}_A,\vec{c}_x)}_\textrm{understandability} + \underbrace{\frac\gamma{L^2} \, \sum_{y=0}^{L^2}d_\mathcal{H}(\vec{c}_A,\vec{c}_y)}_\textrm{discommunication}
$$

---

### Parameters

$\\$

$\alpha$ : Communicative bonus


$\\$

$\gamma$ : Discommunication bonus = 1

$\mu$ : Mutation rate = 0.001

$L$ : System size = 256

$B$ : Length of the bit string = 16



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

### Transition to Diversity

![width:1100px](../fig/jan13/columnScan_lattices_L_256_B_16_mu_0.0001_gamma_1.png)

![width:400px](../fig/jan13/alphaL_number_of_clusters_heatmap_B_16_gamma_1_mu_0.0001.png) $\phantom{mm}$  ![width:400px](../fig/jan13/alphaL_largest_cluster_fraction_heatmap_B_16_gamma_1_mu_0.0001.png)

---

### Different fitness, but coexistence!

$\\$

Boundary fitness is equalized!

$\\$

![width:800px](../fig/jan13/boundary_fitness_dropoff_schematic.png)


---

### Population timeseries


Low $\mu$ $\implies$ punctuated equilibria:
Switching between metastable states.

![width:1000px](../fig/jan13/population_timeseries_L_256_B_16_g_1_a_0.4_mu_1e-05.png)

---

New clusters form at **boundaries!**

$\\$

![width:1000px](../fig/jan13/new_cluster_forming_bndry.png)

---

A **phylogenetic tree** can be drawn.

$\\$

But is it truly correct?

![bg fit right:57%](../fig/jan13/phylogenetictree_L_256_g_1_a_1_B_16_mu_0.001_minSize_10.png)

---

### Emergent Borrowing

1101 and 1110 are equally like to mutate.
Mutants that borrow are selected.

![width:800px](../fig/jan13/schematic_mutation_borrowing.png)

---

![bg fit](../fig/jan13/alpha_mu_raster_lattices_L_256_B_16_gamma_1.png)

---

#### Mutation rate scaling

![width:400px](../fig/jan13/alphaMu_number_of_clusters_heatmap_L_256_B_16_gamma_1.png) $\phantom{m}$ ![width:400px](../fig/jan13/alphaMu_largest_cluster_fraction_heatmap_L_256_B_16_gamma_1.png)

![width:320px](../fig/jan13/mu_scaling_num_clusters_L_256_B_16_gamma_1_loglog.png) $\phantom{mmmm}$ ![width:320px](../fig/jan13/mu_scaling_largest_cluster_L_256_B_16_gamma_1_linlog.png)



---

### 1D Mutation rate scaling

$\\$

![width:1100px](../fig/jan13/1D_mutation_rate_scaling.png)

---

### Why the same size?

Assume boundary fitness equalization & ideal case of 3 languages, $c_1$, $c_2$, $c_3$ with populations $N_1$, $N_2$, $N_3$.


![width:600px](../fig/jan13/boundary_fitness_dropoff_schematic.png)


$$\left(\frac\alpha\gamma - N_3 \right)\left[f_1 - f_2\right] = d_\mathcal{H}(c_2, c_1) \left[N_1 - N_2\right]
$$

---

## Conclusion

$\\$

Robust diversity can arise from antagonism

*Emergent behaviour*: punctuated equilibira, borrowing, coexistence of less-fit languages, new cluster formation at boundaries

Much stems from a **high dimensional state** and a **context-dependent fitness.**