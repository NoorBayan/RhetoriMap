# RhetoriMap

## An Empirical Investigation into Conceptual Mappings and Pragmatic Functions in Classical Arabic Metaphor

**RhetoriMap** is an interdisciplinary research initiative situated at the intersection of **Artificial Intelligence, Cognitive Linguistics, and Quranic Humanities**. The project is fundamentally designed as an empirical investigative framework to explore a critical question in computational rhetoric: *Does human cognitive interpretability inherently translate into computational utility for deep learning models?*

Rather than treating figurative language simply as a detection task, RhetoriMap focuses on the profound rhetorical intent (pragmatics) of metaphors, particularly within highly stylized and morphosyntactically dense texts like the Qur'an.

---

## The Complexity of Quranic Discourse

Classical Arabic, and Quranic discourse in particular, is characterized by extreme rhetorical density and pragmatic polysemy. Metaphors in this text are not mere stylistic decorations; they are intricate conceptual bridges designed to achieve highly specific communicative goals—such as *Warning*, *Affirmation*, or *Glorification*. 

Mapping abstract theological or eschatological concepts to concrete human experiences requires a layered understanding of both the text and the intent behind it. RhetoriMap was built to computationally respect and investigate this profound complexity.

---

## The Theoretical Core

According to cognitive metaphor theories, a metaphor establishes a structured relationship between two distinct domains:

$$M = (S, T)$$

where:
- $S$ represents the **Source Domain** (the concrete concept, e.g., *Trade* or *Light*),
- $T$ represents the **Target Domain** (the abstract concept, e.g., *Guidance* or *Human Behavior*).

RhetoriMap investigates whether explicitly defining this conceptual mapping provides measurable predictive power to determine the ultimate **Pragmatic Function ($P$)** of the verse:

$$(S, T) \xrightarrow{?} P$$

---

## Empirical Validation and Experimental Implementation

To move beyond the theoretical formulation, **RhetoriMap** includes an empirical experimental phase designed to test whether explicitly represented **Source–Target conceptual mappings** provide useful predictive information for identifying the **pragmatic functions** of Classical Arabic metaphors.

The experiments were implemented and executed using **Google Colab**, where the complete computational pipeline was developed and the experimental results were obtained. The notebook includes the implementation of the proposed models, training and evaluation procedures, and the generation of the experimental results used in the empirical investigation.

> 🧪 **Experimental Implementation**  
> The complete experimental notebook and implementation are available on Google Colab:  
> 
> <a href="https://colab.research.google.com/drive/1Zx2PPV5hZe4QOprR3XkyWXTQt0OG_4y-?usp=sharing">
> <img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="32" alt="Google Colab"/>
> </a>
> 
> **[Open the Experimental Notebook](https://colab.research.google.com/drive/1Zx2PPV5hZe4QOprR3XkyWXTQt0OG_4y-?usp=sharing)**

The experimental phase is an essential component of the RhetoriMap methodology because the project does not assume that linguistically motivated conceptual mappings will necessarily improve neural prediction. Instead, their contribution is evaluated empirically by comparing model configurations with and without the explicit conceptual information. This allows the study to distinguish between **theoretical linguistic relevance** and **actual computational utility**.

Accordingly, the experimental results provide the empirical basis for answering the project's central question:

> **Does explicitly representing the conceptual mapping of a Classical Arabic metaphor provide measurable predictive value for identifying its pragmatic function?**

The resulting analysis is therefore intended not merely to report model performance, but to diagnose whether conceptual metaphor information acts as a **complementary signal, a redundant signal, or a potentially interfering signal** within contextual neural representations.

---

## The Research Philosophy: Diagnosis Over SOTA

In the current landscape of Natural Language Processing (NLP), there is a persistent assumption that injecting explicit linguistic rules or structures into neural networks will automatically improve their performance. RhetoriMap challenges this assumption through rigorous empirical diagnosis. 

The project operates on a simple but critical principle:
> *If a rhetorical relationship is theoretically meaningful to human scholars, its actual computational value within a dense neural representation space must be tested, not merely assumed.*

Therefore, RhetoriMap is **not** designed to chase State-of-the-Art (SOTA) accuracy metrics on standard leaderboards. Instead, it serves as a diagnostic environment (an ablation-style framework) to study the interaction—and potential interference—between discrete linguistic categories and the continuous, contextualized embeddings generated by deep language models.

---

## A Bridge Between Three Disciplines

RhetoriMap offers valuable insights across three distinct research communities:

1. **For AI & Machine Learning Researchers:**  
   It provides a testing ground for understanding Multi-Task Learning (MTL) dynamics, inductive biases, and the phenomena of representational interference when fusing symbolic categorical knowledge with deep neural networks.

2. **For Linguists & Cognitive Scientists:**  
   It offers a computational lens to test whether Conceptual Metaphor Theory (CMT) can serve as a robust predictive scaffold for pragmatic meaning construction.

3. **For Humanities & Quranic Studies Scholars:**  
   It introduces a modern, data-driven approach to classical *Balagha* (rhetoric), exploring how ancient semantic relationships can be mapped, measured, and analyzed using contemporary computational methodologies without losing their contextual depth.

---

## The Core Investigative Goal

Ultimately, RhetoriMap seeks to answer:
> **Can computational models benefit from explicitly representing the conceptual mapping of a Classical Arabic metaphor when identifying its pragmatic intent?**

- **If yes**, it suggests that structured human linguistic knowledge acts as a powerful guiding signal for neural networks.
- **If no**, it provides profound insights into the architecture of modern language models, suggesting that explicitly forcing discrete human concepts might create bottlenecks that conflict with the models' native contextual understanding.

In either scenario, RhetoriMap bridges the gap between traditional Arabic rhetorical scholarship and the frontier of representation learning.

---

## RhetoriMap in One Sentence

> **RhetoriMap is a diagnostic computational framework that investigates whether the structured conceptual mappings underlying highly complex Classical Arabic metaphors provide complementary predictive signals for understanding their pragmatic functions.**
