## Deriving an equation based on boundary fitness being equalized

We have $M$ languages: $c_1$, $c_2$, and $c_3 \dots c_{M}$. A language $c_i$ has a population (number of speakers) $N_i$, and the bitstring consists of $f_i$ ones. Without loss of generality, we consider the boundary between $c_1$ and $c_2$. If the fitness at the boundary is equal, then,

$$
\begin{align*}
    \textrm{Global fitness of 1} + \textrm{Local fitness of 1} = \textrm{Global fitness of 2} + \textrm{Local fitness of 2} \\
    \frac\gamma{B} \left[ N_1 d_\mathcal{H}(c_1, c_1) + N_2 d_\mathcal{H}(c_1, c_2) + \sum_{i=3}^N N_i d_\mathcal{H}(c_1, c_i) \right] + \frac\alpha{B}\left[U(c_1,c_1) + U(c_1, c_2)\right] \\
    = \frac\gamma{B} \left[ N_1 d_\mathcal{H}(c_2, c_1) + N_2 d_\mathcal{H}(c_2, c_2) + \sum_{i=3}^N N_i d_\mathcal{H}(c_2, c_i) \right] + \frac\alpha{B}\left[U(c_2,c_1) + U(c_2, c_2)\right] \\
\end{align*}
$$

Using the relations:
$$
\begin{align*}
    U(c_i, c_i) = f_i \\
    d_\mathcal{H}(c_i, c_i) = 0
\end{align*}
$$

$$
\begin{align*}
    \gamma \left[ N_2 d_\mathcal{H}(c_1, c_2) + \sum_{i=3}^N N_i d_\mathcal{H}(c_1, c_i) \right] + \alpha\left[f_1 + U(c_1, c_2)\right] \\
    = \gamma \left[ N_1 d_\mathcal{H}(c_2, c_1) + \sum_{i=3}^N N_i d_\mathcal{H}(c_2, c_i) \right] + \alpha\left[U(c_2,c_1) + f_2\right] \\
\end{align*}
$$
$$
\begin{align*}
    \alpha\left[f_1 + U(c_1, c_2) - U(c_2,c_1) - f_2\right] 
    = \gamma \left[ N_1 d_\mathcal{H}(c_2, c_1)+ \sum_{i=3}^N N_i d_\mathcal{H}(c_2, c_i) - N_2 d_\mathcal{H}(c_1, c_2) - \sum_{i=3}^N N_i d_\mathcal{H}(c_1, c_i) \right]
\end{align*}
$$
$$
\begin{equation}
\frac\alpha\gamma\left[f_1 - f_2\right] = d_\mathcal{H}(c_2, c_1) \left[N_1 - N_2\right]+ \sum_{i=3}^N N_i \left[d_\mathcal{H}(c_2, c_i) - d_\mathcal{H}(c_1, c_i)\right] 
\end{equation}
$$
Let us consider a simplified case where $M=3$, and the last language $c_3$ consists of only $1$s. Then, $d_\mathcal{H}(c_i, c_3) = B - f_i$

$$
\begin{align*}
\frac\alpha\gamma\left[f_1 - f_2\right] = d_\mathcal{H}(c_2, c_1) \left[N_1 - N_2\right]+ N_3 \left[d_\mathcal{H}(c_2, c_3) - d_\mathcal{H}(c_1, c_3)\right]      \\
\frac\alpha\gamma\left[f_1 - f_2\right] = d_\mathcal{H}(c_2, c_1) \left[N_1 - N_2\right]+ N_3 \left[(B - f_2) - (B - f_1)\right]      \\
\end{align*}
$$
$$
\begin{equation}
\left(\frac\alpha\gamma - N_3 \right)\left[f_1 - f_2\right] = d_\mathcal{H}(c_2, c_1) \left[N_1 - N_2\right]
\end{equation}
$$


## Trying to simultaneously equalize *all* boundaries:

We know that for the fitness between $c_1$ and $c_2$ to be equalized, we have

$$
\begin{equation}
\frac\alpha\gamma\left[f_1 - f_2\right] = d_\mathcal{H}(c_2, c_1) \left[N_1 - N_2\right]+ \sum_{i=3}^N N_i \left[d_\mathcal{H}(c_2, c_i) - d_\mathcal{H}(c_1, c_i)\right] 
\end{equation}
$$

Without loss of generality, this can be written for two neighbours $c_k$ and $c_{k+1}$:

$$
\frac\alpha\gamma\left[f_k - f_{k+1}\right] = d_\mathcal{H}(c_k, c_{k+1}) \left[N_k - N_{k+1}\right] - \sum_{i=1,\; i\neq k, k+1}^N N_i \left[d_\mathcal{H}(c_k, c_i) - d_\mathcal{H}(c_{k+1}, c_i)\right] 
$$

Now, we have $N$ of such equations (1 for each $k\in[1, N]$). Summing them up, the left term will cancel, as $\alpha$ and $\gamma$ are constants. We're left with

$$
\sum_{k=1}^N d_\mathcal{H}(c_k, c_{k+1}) \left[N_k - N_{k+1}\right] = \sum_{k=1}^N \quad \sum_{i=1,\; i\neq k, k+1}^N N_i \left[d_\mathcal{H}(c_k, c_i) - d_\mathcal{H}(c_{k+1}, c_i)\right] 
$$

Considering the left term first:

$$
\sum_{k=1}^N d_\mathcal{H}(c_k, c_{k+1}) \left[N_k - N_{k+1}\right]
$$
It can be split into two:
$$
\sum_{k=1}^N d_\mathcal{H}(c_k, c_{k+1}) N_k - \sum_{k=1}^N d_\mathcal{H}(c_k, c_{k+1})  N_{k+1}  \\
\sum_{k=1}^N d_\mathcal{H}(c_k, c_{k+1}) N_k - \sum_{j=2}^{N+1} d_\mathcal{H}(c_{j-1}, c_{j})  N_{j} \\
\sum_{k=1}^N d_\mathcal{H}(c_k, c_{k+1}) N_k - \sum_{k=1}^{N} d_\mathcal{H}(c_{k-1}, c_{k})  N_{k} \\
\sum_{k=1}^N N_k \left( d_\mathcal{H}(c_k, c_{k+1}) -  d_\mathcal{H}(c_{k-1}, c_{k})\right)

$$

....may be continued.
