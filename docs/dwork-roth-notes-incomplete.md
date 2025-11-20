## 1: The promise of differential privacy

- The goal of DP is to ensure that the impact on an individual is the same regardless of whether or not they participate in the dataset.
  - That is, any impacts that happen to an individual are due to conclusions reached from the data rather than them being in the data themselves (complete independence).
  - Any sequence of outputs (responses to a query) is "essentially" equally likely to occur regardless of whether the individual exists in the dataset. The probabilities here are taken over random choices made by our privacy mechanism, and the term "essentially" is captured by a parameter $\varepsilon$. A small $\varepsilon$ is desirable to yield better privacy (with the caveat of less accurate responses).
- DP is a definition, rather than an algorithm. Multiple algorithms exist for any task $T$ such that it achieves $T$ in a $\varepsilon$-differentially private manner.

### 1.1: Privacy-preserving data analysis

- Rich datasets cannot be fully anonymized by simply removing identifiers, but differential privacy protects individuals even when adversaries have auxiliary information (e.g. when they try to perform a "linkage attack").
  - Re-identification isn't the only privacy risk. Even if a specific individual's data record can't be linked to them, sensitive information about them can still be inferred from aggregated or "anonymized" data. Simply hiding identities does not prevent privacy harm.
- Even if it were feasible to identify if a query is trying to single out an individual, it would not be effective because we could just perform a "differencing attack." We could query for "people with disease A" and "people (not named X) with disease A" and reverse the traits of person X.
  - You also can't try to audit and check for differencing attacks by seeing if any series of queries might constitute one. Refusing a query can itself be disclosive. It also is completely infeasible.
- Summary statistics cannot guarantee privacy, since if we ask enough specific questions via query we can effectively "reconstruct" an entire individual's participation in the dataset.

## 2: Basic terms

### 2.1: The model of computation

- We assume the existence of a *curator* that is trustworthy. This curator holds a database $D$ containing $n$ rows of data pertaining to $n$ individuals.
  - In the *offline*/*non-interactive* model, the curator produces some kind of sanitized object that is meant to allow for action on $D$ but is not $D$ itself. From there, the curator no longer plays any role in the analysis and can destroy the original $D$.
  - In the *online*/*interactive* model, the curator is replaced by a protocol that allows for queries to be sent by an analyst, and for responses to be received.
    - This poses the problem that we don't know the intention of the analyst. We want to provide responses to all possible queries, but privacy will deterorate as we do so since we reveal more and more information (recall the *Fundamental Law of Information Recovery*). Thus, necessasrily, accuracy should deteriorate with the number of queries.
- A *privacy mechanism* (or simply a *mechanism*) is an algorithm that has:
  - Inputs:
    - $D$: the database in interest
    - $\mathcal{X}$: a universe of data types, or the set of all possible database rows
    - Random bits
    - Optionally, a set of queries
  - Outputs:
    - An output string, which can be many things in this case:
      - When the set of queries does not exist, we are in the non-interactive model and thus we should output a synthetic database, a sanitized database, or a summary statistic. The hope is that this output string should answer future queries with good accuracy.
      - When the set of queries does exist, we are in the interactive model and thus we should output a response to each query with hopefully good accuracy.

### 2.2: Towards defining private data analysis

- Although we naturally want to protect privacy by adopting the "nothing is learned about any individual" approach, this is unachievable due to the fact that an analyst infers information from the individual based on the entire dataset. We instead seek a guarantee that database access doesn't significantly change what can be inferred about anyone compared to what could be inferred with only auxiliary information.
  - In the Alice-Bob-Eve world often presented in cryptography, Alice is the sender, Bob is the receiver, and Eve is the eavesdropper. This analogy doesn't really work with data analysis since Bob and Eve have the potential to be the same person, and thus denying Eve information will also deny Bob information.

### 2.3: Formalizing differential privacy

- Randomness is a good source of privacy. An early example is people being surveyed about something embarassing. The algorithm is as such:
  1. Flip a coin.
  2. If tails, respond truthfully.
  3. If heads, flip a second coin and respond "Yes" if heads, "No" if tails.
  - Here, the "privacy" comes from plausible deniability, since even if someone did truthfully respond "Yes" then they can claim that they just flipped two coins and got two heads.
  - Deterministic mechanisms allow outputs that can always distinguish between neighboring databases (that differ by a single row, which would be an individual's data). Adding randomness prevents perfect inference.
- We define a randomized algorithm $\mathcal{M}$ as a mapping $\mathcal{M} \colon A \rightarrow \Delta(B)$ for some domain $A$ and range $\Delta(B)$ (where $\Delta(B)$ is the set of all probability distributions over $B$, with the requirement that its numbers sum to 1).
  - Rather than seeing databases $x$ as a collection of records from universe $\mathcal{X}$, we can instead see them as histograms $x \in \mathbb{N}^{|\mathcal{X}|}$ (where $\mathbb{N}$ is the set of non-negative integers, including 0) in which each $x_i$ represents the number of times that type $i \in \mathcal{X}$ appears in the database. For instance, if your universe is $\{\text{student}, \text{professor}, \text{staff}\}$, then a database $x$ could be $(150, 20, 30)$.
    - With this, we can compute the distance between two databases $x$ and $y$ as the $\ell_1$ norm of the difference between the two histograms: $\|x - y\|_1 = \sum_{i=1}^{|\mathcal{X}|} |x_i - y_i|$. We also note that simply $\|x\|_1 = \sum_{i=1}^{|\mathcal{X}|} x_i$ describes the size of the database.
- We can now formally define differential privacy. A randomized algorithm $\mathcal{M}$ with domain $\mathbb{N}^{|\mathcal{X}|}$ is $(\varepsilon, \delta)$-differentially private if, for all $S \subseteq \text{Range}(\mathcal{M})$ and for all $x, y \in \mathbb{N}^{|\mathcal{X}|}$ such that $\|x - y\|_1 \leq 1$:
  $$
  \Pr[\mathcal{M}(x) \in S] \leq e^{\varepsilon} \Pr[\mathcal{M}(y) \in S] + \delta
  $$
  where the probability is taken over the random choices (coin flips) made by $\mathcal{M}$. If $\delta = 0$, we say that $\mathcal{M}$ is simply $\varepsilon$-differentially private. Breaking this down:

  - "$x, y \in \mathbb{N}^{|\mathcal{X}|}$ such that $\|x - y\|_1 \leq 1$": $x$ and $y$ represent two databases (or dataset histograms) that differ in only one individual's data. This can be called a "neighboring database."
  - "$S \subseteq \text{Range}(\mathcal{M})$": $S$ is any possible set of outputs that the mechanism $\mathcal{M}$ could produce.
  - "$\Pr[\mathcal{M}(x) \in S]$", "$\Pr[\mathcal{M}(y) \in S]$": The probability (over the randomization of $\mathcal{M}$) that applying $\mathcal{M}$ to database $x$ / $y$ produces an output in the set $S$.
  - "$e^{\varepsilon}$ (or $\exp(\varepsilon)$)" This is a multiplicative bound on how much more likely any particular output (or set of outputs) could occur because one individual's data was included or not. The smaller $\varepsilon$, the closer the probabilities must be.
  - "$+ \delta$": $\delta$ is an "additive slack" parameter. With probability at most $\delta$, the output of $\mathcal{M}$ is allowed to violate the multiplicative $e^\varepsilon$ bound. Ideally, $\delta$ is very small, usually negligible.
    - We typically look for $\delta$ to be less than the inverse of any polynomial in the size of the database. $\delta = 1/\|x\|_1$ is very dangerous since it permits "just a few" violations.
    - Even when $\delta$ is negligible, there are important differences between $(\varepsilon, 0)$- and $(\varepsilon, \delta)$-DP:
      - $(\varepsilon, 0)$-DP ensures that, for any possible output, the probability of that output is nearly the same for all neighboring databases.
      - $(\varepsilon, \delta)$-DP allows for an extremely small probability $\delta$ that this similarity fails. In the failure case, given an output $\xi \sim \mathcal{M}(x)$, we are able to find a database $y$ such that $\xi$ is much more likely to be produced on $y$ than $x$ (the mass of $\xi$ in $\mathcal{M}(y)$ is much larger than the mass of $\xi$ in $\mathcal{M}(x)$).
        - With this, we define a quantity known as *privacy loss* induced by observing $\xi$:
          $$
          \mathcal{L}_{\mathcal{M}(x) \parallel \mathcal{M}(y)} = \ln \left( \frac{\Pr[\mathcal{M}(x) \in S]}{\Pr[\mathcal{M}(y) \in S]} \right)
          $$
          This quantity is positive when $x$ is more likely than $y$ to produce $\xi$, and negative when $y$ is more likely than $x$ to produce $\xi$. As we will see later, the absolute value of this privacy loss will be bounded by $\varepsilon$ with at least probability $1 - \delta$ within $(\varepsilon, \delta)$-DP.
- DP is immune to post-processing: no one can make the output less private by applying any function to it after release, as long as they don't have extra knowledge about the database (applying any data-independent mapping to the output of a differentially private mechanism cannot increase privacy loss).
- From the definition of DP we can also derive *composite privacy* and *group privacy*:
  - Combining $k$ $(\varepsilon_i, \delta_i)$-DP mechanisms gives $(\sum_i \varepsilon_i, \sum_i \delta_i)$-DP overall for $1 \leq i \leq k$.
  - For a group of size $k$, $(\varepsilon, 0)$-DP guarantees $(k\varepsilon, 0)$-DP; privacy degrades linearly in group size.

### 2.3.1/2: What differential privacy promises/does not promise

- DP guarantees that the risk to any individual is almost the same whether or not their data is included in the dataset, by carefully bounding how much the output distribution can change (i.e. $\text{exp}(\varepsilon) \approx (1 + \varepsilon)$ multiplicatively). In other words, participation should not significantly increase the chance of any harm or unwanted outcome for that individual.
- DP only limits what can be learned from an individual's participation, not what can be inferred from population-level statistics or correlations present regardless of their inclusion. If a survey teaches an analyst that specific private attributes are correlated with a publicly observable attributes, this is not a violation of DP, since this same correlation would exist independent of the presence/absence of the individual.

### 2.3.2: Final remarks on the definition

- Granularity is important. The level of differential privacy, whether applied to individuals, edges, or events, should be explicitly specified, as the privacy guarantee depends entirely on what is considered a "single entry" in the database.
- There are big differences between small and large $\varepsilon$ values:
  - Small $\varepsilon$ means outputs from neighboring databases are nearly indistinguishable. Small $\varepsilon$ values all have similar behavior; you could fail on $(\varepsilon, 0)$-DP but succeed on $(2\varepsilon, 0)$-DP, both of which are still very strong guarantees.
  - Large $\varepsilon$ is a very weak guarantee, but what exactly it "allows" can range widely: the privacy loss could be small (so little is actually leaked), or catastrophic (so much is leaked that almost the entire database could be reconstructed).
- Recall that a privacy mechanism can occasionally take on auxiliary parameters $w$ as input in addition to the database $x$.
  - For example, $w$ may specify a query $q_w$ on $x$, or a collection $\mathcal{Q}_w$ of queries on $x$. The mechanism $\mathcal{M}(w, x)$ might (respectively) respond with a DP approximation to $q_w(x)$ or some/all of the outputs to the queries in $\mathcal{Q}_w$.
    - We can define $Q_w = \{q \colon \mathcal{X}^n \rightarrow \mathbb{R}\}$; in this case, $\mathcal{M}$ is called a *synopsis generator*. It produces a DP synopsis $\mathcal{A}$ from which a reconstruction procedure $R(\mathcal{A}, v)$ (where $v$ specifies a query $q_v \in Q_w$, where sometimes we abuse notation and write $R(\mathcal{A}, q)$) can usefully approximate answers to queries in $Q_w$.
      - A synthetic database is a special case of a synopsis, since each row of the synthetic database is of the same type as each row of the original database. We can perform the same operations using the same software, and thus we remove the need for $R$.
  - We may also include a *security parameter* $\kappa$ that governs how small $\delta = \delta(\kappa)$ can be ($\mathcal{M}(\kappa, \cdot)$ must be $(\varepsilon, \delta(\kappa))$-DP). We will require that $\delta$ be a negligible function in $\kappa$, i.e. $\delta = \kappa^{-\omega(1)}$. Thus, $\delta$ should be "cryptographically small" while $\varepsilon$ can be "moderately small."

## 3: Basic techniques and composition theorems

### 3.1: Useful probabilistic tools

- *Additive Chernoff bound:* If you take the average of bounded, independent random variables, the probability that the average deviates by more than $\varepsilon$ from its expectation drops off exponentially fast in $m\varepsilon^2$. In other words, it's extremely unlikely for the average to be far from the mean if you have many samples.
- *Multiplicative Chernoff bound:* Similar to additive Chernoff, but here the deviation is measured as a fraction of the mean (i.e., the probability that the sample mean is, say, 20% higher or lower than $\mu$) and the tail probability decays exponentially in $m\mu\varepsilon^2$. This is particularly useful when $\mu$ is small.
- *Azuma's inequality:* When the random variables are not independent but form a "martingale" (think: each step can only change the expected value by a bounded amount), Azuma's inequality provides a similar guarantee—large deviations from the expected value are still extremely unlikely, with the 'strength' of the bound depending on how much each variable can affect the outcome.
- *Stirling's approximation:* For large $n$, the factorial $n!$ grows very quickly, but Stirling's formula tells us that $n! \approx \sqrt{2\pi n}(n/e)^n$. This is handy for estimating probabilities or combinatorial quantities when working with big numbers.
