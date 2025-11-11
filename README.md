# DoRA: Weight-<i>D</i>ecomposed L<i>o</i>w-<i>R</i>ank <i>A</i>daptation
### Authors: Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen

**Citation:**
Liu, Shih-Yang, Chien-Yi Wang, Hongxu Yin, et al. “DoRA: Weight-Decomposed Low-Rank Adaptation.” arXiv:2402.09353. Preprint, arXiv, July 9, 2024. https://doi.org/10.48550/arXiv.2402.09353. </br>
Published at the 41st International Conference on Machine Learning (ICML), 2024.

### Presentor: Suyash Deshmukh

---

## 1. Paper Overview

[Five-minute overview providing context, stating the problem the paper is addressing, characterizing the approach, and giving a brief account of how the problem was addressed.]: #

Large pre-trained models are powerful but costly to fine-tune.
Full fine-tuning (FT) updates every parameter — giving strong results but demanding huge compute.
LoRA (Low-Rank Adaptation) became a popular alternative by inserting small low-rank adapters so that only a fraction of weights are trained, with no extra inference cost.

However, despite LoRA’s efficiency, there’s still a consistent accuracy gap between LoRA and full fine-tuning.
Most prior work simply assumed this was due to LoRA’s limited number of trainable parameters.
The authors of DoRA wanted to investigate this assumption instead of accepting it — to ask why LoRA behaves differently.

[Question] 
<details>
    <summary>
        Answer
    </summary>
  [Answer]
</details>

Question:
If you wanted to analyze what’s changing inside a matrix of weights — and you remember that matrices are made of vectors — how could you decompose it?
(Hint: scalars have magnitude; what’s special about a vector?)

That’s right — you can decompose vectors into a magnitude and a direction.
And that’s exactly what their analysis focuses on, and what DoRA does differently from LoRA.

DoRA decomposes each pre-trained weight W into a magnitude m and direction V.
- Direction is updated efficiently using LoRA’s low-rank matrices.
- Magnitude is updated directly (a simple vector).

Merges both parts back after training → no extra inference cost.

This separation lets DoRA perform more flexible and fine-grained updates — LoRA tended to scale both parts together, while FT could adjust them independently.

Outperforms LoRA while retaining the efficiency.

> [!Note]
> Extra Information

---

## 2. Context and Motivation
- Background on **PEFT methods** (Parameter-Efficient Fine-Tuning)
- Problem statement: why LoRA still lags behind full fine-tuning
- Objective: bridge the performance gap without adding inference cost

---

## 3. Method Summary
- Key idea behind **weight decomposition** (magnitude + direction)
- How DoRA modifies LoRA’s approach
- Benefits in learning stability and fine-tuning efficiency

---

## 4. Architecture Overview
- Formal **pseudocode description** of DoRA
- Clear comparison vs. LoRA and Full Fine-Tuning
- Mathematical formulation (Eq. 5 from paper)
- Diagram of weight decomposition (Figure 1)

<p>
Below is a <strong>side-by-side formal pseudocode comparison</strong> between <b>LoRA</b> and <b>DoRA</b>.
</p>

<table>
<tr>
<td width="50%" valign="top">

<h3>Algorithm 1: Low-Rank Adaptation (LoRA)</h3>

<hr>

<p><b>Input:</b><br>

Pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$,  
training data $\mathcal{D} = \{(x_i, y_i)\}$,  
rank $r \ll \min(d, k)$,  
learning rate $\eta$.
</p>

<hr>

<p><b>Parameters:</b><br>

Trainable low-rank matrices $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$.
</p>

<hr>

<ol>
<li>

Initialize $A \leftarrow Uniform Kaiming Distribution$, $B \leftarrow 0$.</li>
<li>

Repeat for each minibatch $(x, y) \in \mathcal{D}$:</li>
<ul>
<li>

Compute low-rank update: $\Delta W \leftarrow B A$</li>
<li>

Form adapted weight: $W \leftarrow W_0 + \Delta W$</li>
<li>

Forward pass: $\hat{y} \leftarrow f(x; W)$</li>
<li>

Compute loss: $\mathcal{L} \leftarrow \ell(\hat{y}, y)$</li>
<li>

Backpropagate gradients w.r.t. $A, B$</li>
<li>

Update parameters:<br>
$A \leftarrow A - \eta \nabla_A \mathcal{L}$,  
$B \leftarrow B - \eta \nabla_B \mathcal{L}$</li>
</ul>
<li>

Until convergence or max epochs reached.</li>
<li>

Return $W = W_0 + B A$.</li>
</ol>
<hr>

<p><b>

Memory Complexity:</b> $\mathcal{O}(r(d + k))$</p>

</td>

<td width="50%" valign="top" style="border-left:1px solid #ccc; padding-left:15px;">

<h3>Algorithm 2: Weight-Decomposed Low-Rank Adaptation (DoRA)</h3>

<hr>

<p><b>Input:</b><br>

Pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$,  
training data $\mathcal{D} = \{(x_i, y_i)\}$,  
rank $r \ll \min(d, k)$,  
learning rate $\eta$.
</p>

<hr>

<p><b>Parameters:</b><br>

Magnitude vector $m = \|W_0\|_c \in \mathbb{R}^{k}$;<br>
Direction matrix $V = \frac{W_0}{\|W_0\|_c} \in \mathbb{R}^{d \times k}$;<br> Trainable magnitude correction $\Delta m \in \mathbb{R}^{k}$;<br>
Low-rank trainable matrices $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$.<br>
</p>

<hr>

<ol>
<li>

Initialize $A \leftarrow Uniform Kaiming Distribution$, $B \leftarrow 0$, $\Delta m \leftarrow 0$.</li>
<li>

Compute base decomposition:<br>
$m \leftarrow \|W_0\|_c$,  
$V \leftarrow W_0 / m$.</li>
<li>

Repeat for each minibatch $(x, y) \in \mathcal{D}$:</li>
<ul>
<li>

Compute low-rank directional update: $\Delta V \leftarrow B A$</li>
<li>

Form adapted weight: $W \leftarrow (m + \Delta m) \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c}$</li>
<li>

Forward pass: $\hat{y} \leftarrow f(x; W)$</li>
<li>

Compute loss: $\mathcal{L} \leftarrow \ell(\hat{y}, y)$</li>
<li>

Backpropagate gradients w.r.t. $A, B$, $\Delta m$.</li>
<li>

Update parameters:<br>
$A \leftarrow A - \eta \nabla_A \mathcal{L}$,  
$B \leftarrow B - \eta \nabla_B \mathcal{L}$,<br>
$\Delta m \leftarrow \Delta m - \eta \nabla_{\Delta m} \mathcal{L}$</li>
</ul>
<li>

Until convergence or max epochs reached.</li>
<li>

Return $W = (m + \Delta m) \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c}$.</li>
</ol>
<hr>

<p><b>

Memory Complexity:</b> $\mathcal{O}(r(d + k) + k)$</p>

</td>
</tr>
</table>

<br>

<p><b>Summary:</b><br>
DoRA extends LoRA by decomposing pretrained weights into <em>magnitude</em> and <em>direction</em>, updating them separately. 
This decoupling allows DoRA to reproduce full fine-tuning’s flexibility while preserving LoRA’s efficiency and mergeability.
</p>

---

## 5. Experimental Results
- Summary of main experiments:
  - Commonsense reasoning (LLaMA models)
  - Vision-language tasks (VL-BART, LLaVA)
  - Compatibility with VeRA (DVoRA)
- Key performance tables and figures
- Highlights of improvements (accuracy %, parameter reduction, etc.)

---

## 6. Critical Analysis
- Strengths of DoRA
- Limitations or potential weaknesses
- What could have been analyzed further or validated more thoroughly
- Open questions raised by this paper

---

## 7. Impact and Relevance
- Broader impact on AI fine-tuning
- Implications for LLMs, LVLMs, and efficient adaptation
- Future extensions (e.g., QDoRA, audio applications)
- How it changes the landscape of PEFT research

---

## 8. Code Demonstration (Optional)
- Short Python snippet or pseudocode showing DoRA integration  
- Link to official [NVLabs/DoRA GitHub repository](https://github.com/NVlabs/DoRA)

---

## 9. References & Resource Links
Include full citations for all external works mentioned above (LoRA, VeRA, QLoRA, etc.)
1. [Official Paper (ICML 2024)](https://github.com/NVlabs/DoRA)
2. [DoRA GitHub Repository](https://github.com/NVlabs/DoRA)
3. [LoRA Original Paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
4. [VeRA Paper (Kopiczko et al., 2024)](https://arxiv.org/abs/2402.10362)
5. [Sebastian Raschka’s DoRA Tutorial](https://sebastianraschka.com/blog/2024/dora.html)