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

## 🏗️ Architecture Overview

Below are the formal algorithms for **LoRA** and **DoRA**.

---
<table>
<tr>
<th>LoRA</th>
<th>DoRA</th>
</tr>
<tr>
<td>
  
```
**Input:**  
Pretrained weight matrix \( W_0 \in \mathbb{R}^{d \times k} \),  
training data \( \mathcal{D} = \{(x_i, y_i)\} \),  
rank \( r \ll \min(d, k) \),  
learning rate \( \eta \).

**Parameters:**  
- Trainable low-rank matrices:  
  \( A \in \mathbb{R}^{r \times k} \), \( B \in \mathbb{R}^{d \times r} \)  
- Fixed pretrained weight: \( W_0 \)

---

**Algorithm:**

1. **Initialize** \( A \leftarrow 0 \), \( B \leftarrow 0 \)  
   \( \triangleright \) Ensure pretrained output is unchanged at start.
2. **Repeat** for each minibatch \( (x, y) \in \mathcal{D} \):
3. &emsp; **Compute** low-rank update:  
   &emsp; \( \Delta W \leftarrow B A \)
4. &emsp; **Form adapted weight:**  
   &emsp; \( W \leftarrow W_0 + \Delta W \)
5. &emsp; **Forward pass:**  
   &emsp; \( \hat{y} \leftarrow f(x; W) \)
6. &emsp; **Compute loss:**  
   &emsp; \( \mathcal{L} \leftarrow \ell(\hat{y}, y) \)
7. &emsp; **Backpropagate gradients** w.r.t. \( A, B \)
8. &emsp; **Update parameters:**  
   &emsp; \( A \leftarrow A - \eta \, \nabla_A \mathcal{L} \)  
   &emsp; \( B \leftarrow B - \eta \, \nabla_B \mathcal{L} \)
9. **Until** convergence or max epochs reached  
10. **Return** \( W = W_0 + B A \)

---

**Remarks:**  
- Only \( A \) and \( B \) are trainable; \( W_0 \) remains frozen.  
- \( \Delta W = B A \) is rank-\( r \), enabling efficient adaptation.  
- After fine-tuning, \( W_0 \) and \( \Delta W \) merge for inference with no extra cost.  
```
  
</td>
<td>

```
**Input:**  
Pretrained weight matrix \( W_0 \in \mathbb{R}^{d \times k} \),  
training data \( \mathcal{D} = \{(x_i, y_i)\} \),  
rank \( r \ll \min(d, k) \),  
learning rate \( \eta \).

**Parameters:**  
- <span style="color:green">Magnitude vector \( m = \|W_0\|_c \in \mathbb{R}^{k} \)</span>  
- <span style="color:green">Direction matrix \( V = \frac{W_0}{\|W_0\|_c} \in \mathbb{R}^{d \times k} \)</span>  
- <span style="color:green">Trainable magnitude correction \( \Delta m \in \mathbb{R}^{k} \)</span>  
- Low-rank trainable matrices \( A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r} \)

---

**Algorithm:**

1. **Initialize**  
   \( A \leftarrow 0, \ B \leftarrow 0, \ <span style="color:green">\Delta m \leftarrow 0</span> \)
2. **Compute base decomposition:**  
   <span style="color:green">\( m \leftarrow \|W_0\|_c \)</span>  
   <span style="color:green">\( V \leftarrow W_0 / m \)</span>
3. **Repeat** for each minibatch \( (x, y) \in \mathcal{D} \):
4. &emsp; **Compute low-rank directional update:**  
   <span style="color:green">\( \Delta V \leftarrow B A \)</span>
5. &emsp; **Form adapted weight:**  
   <span style="color:green">\( W \leftarrow (m + \Delta m) \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c} \)</span>
6. &emsp; **Forward pass:**  
   \( \hat{y} \leftarrow f(x; W) \)
7. &emsp; **Compute loss:**  
   \( \mathcal{L} \leftarrow \ell(\hat{y}, y) \)
8. &emsp; **Backpropagate gradients** w.r.t. \( A, B, <span style="color:green">\Delta m</span> \)
9. &emsp; **Update parameters:**  
   \( A \leftarrow A - \eta \, \nabla_A \mathcal{L} \)  
   \( B \leftarrow B - \eta \, \nabla_B \mathcal{L} \)  
   <span style="color:green">\( \Delta m \leftarrow \Delta m - \eta \, \nabla_{\Delta m} \mathcal{L} \)</span>
10. &emsp; **(Optional optimization):**  
    <span style="color:green">Detach gradient through normalization  
    \( \nabla_{\|V + \Delta V\|_c} \leftarrow 0 \)</span>
11. **Until** convergence or max epochs reached  
12. **Return**  
    <span style="color:green">\( W = (m + \Delta m) \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c} \)</span>
```

</td>
</tr>
</table>

<details open>
<summary><strong>Algorithm 1: Low-Rank Adaptation (LoRA)</strong></summary>

**Input:**  
Pretrained weight matrix \( W_0 \in \mathbb{R}^{d \times k} \),  
training data \( \mathcal{D} = \{(x_i, y_i)\} \),  
rank \( r \ll \min(d, k) \),  
learning rate \( \eta \).

**Parameters:**  
- Trainable low-rank matrices:  
  \( A \in \mathbb{R}^{r \times k} \), \( B \in \mathbb{R}^{d \times r} \)  
- Fixed pretrained weight: \( W_0 \)

---

**Algorithm:**

1. **Initialize** \( A \leftarrow 0 \), \( B \leftarrow 0 \)  
   \( \triangleright \) Ensure pretrained output is unchanged at start.
2. **Repeat** for each minibatch \( (x, y) \in \mathcal{D} \):
3. &emsp; **Compute** low-rank update:  
   &emsp; \( \Delta W \leftarrow B A \)
4. &emsp; **Form adapted weight:**  
   &emsp; \( W \leftarrow W_0 + \Delta W \)
5. &emsp; **Forward pass:**  
   &emsp; \( \hat{y} \leftarrow f(x; W) \)
6. &emsp; **Compute loss:**  
   &emsp; \( \mathcal{L} \leftarrow \ell(\hat{y}, y) \)
7. &emsp; **Backpropagate gradients** w.r.t. \( A, B \)
8. &emsp; **Update parameters:**  
   &emsp; \( A \leftarrow A - \eta \, \nabla_A \mathcal{L} \)  
   &emsp; \( B \leftarrow B - \eta \, \nabla_B \mathcal{L} \)
9. **Until** convergence or max epochs reached  
10. **Return** \( W = W_0 + B A \)

---

**Remarks:**  
- Only \( A \) and \( B \) are trainable; \( W_0 \) remains frozen.  
- \( \Delta W = B A \) is rank-\( r \), enabling efficient adaptation.  
- After fine-tuning, \( W_0 \) and \( \Delta W \) merge for inference with no extra cost.  

**Memory Complexity:** \( O(r(d + k)) \)
</details>

---

<details open>
<summary><strong>Algorithm 2: Weight-Decomposed Low-Rank Adaptation (DoRA)</strong></summary>

**Input:**  
Pretrained weight matrix \( W_0 \in \mathbb{R}^{d \times k} \),  
training data \( \mathcal{D} = \{(x_i, y_i)\} \),  
rank \( r \ll \min(d, k) \),  
learning rate \( \eta \).

**Parameters:**  
- <span style="color:green">Magnitude vector \( m = \|W_0\|_c \in \mathbb{R}^{k} \)</span>  
- <span style="color:green">Direction matrix \( V = \frac{W_0}{\|W_0\|_c} \in \mathbb{R}^{d \times k} \)</span>  
- <span style="color:green">Trainable magnitude correction \( \Delta m \in \mathbb{R}^{k} \)</span>  
- Low-rank trainable matrices \( A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r} \)

---

**Algorithm:**

1. **Initialize**  
   \( A \leftarrow 0, \ B \leftarrow 0, \ <span style="color:green">\Delta m \leftarrow 0</span> \)
2. **Compute base decomposition:**  
   <span style="color:green">\( m \leftarrow \|W_0\|_c \)</span>  
   <span style="color:green">\( V \leftarrow W_0 / m \)</span>
3. **Repeat** for each minibatch \( (x, y) \in \mathcal{D} \):
4. &emsp; **Compute low-rank directional update:**  
   <span style="color:green">\( \Delta V \leftarrow B A \)</span>
5. &emsp; **Form adapted weight:**  
   <span style="color:green">\( W \leftarrow (m + \Delta m) \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c} \)</span>
6. &emsp; **Forward pass:**  
   \( \hat{y} \leftarrow f(x; W) \)
7. &emsp; **Compute loss:**  
   \( \mathcal{L} \leftarrow \ell(\hat{y}, y) \)
8. &emsp; **Backpropagate gradients** w.r.t. \( A, B, <span style="color:green">\Delta m</span> \)
9. &emsp; **Update parameters:**  
   \( A \leftarrow A - \eta \, \nabla_A \mathcal{L} \)  
   \( B \leftarrow B - \eta \, \nabla_B \mathcal{L} \)  
   <span style="color:green">\( \Delta m \leftarrow \Delta m - \eta \, \nabla_{\Delta m} \mathcal{L} \)</span>
10. &emsp; **(Optional optimization):**  
    <span style="color:green">Detach gradient through normalization  
    \( \nabla_{\|V + \Delta V\|_c} \leftarrow 0 \)</span>
11. **Until** convergence or max epochs reached  
12. **Return**  
    <span style="color:green">\( W = (m + \Delta m) \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c} \)</span>

---

**Remarks:**  
- <span style="color:green">Separates magnitude and direction updates for finer control.</span>  
- <span style="color:green">Uses low-rank updates only for direction (LoRA-style) and direct update for magnitude.</span>  
- <span style="color:green">Gradient detachment reduces training memory with minimal loss.</span>  
- Mergeable at inference — same runtime cost as LoRA.  

**Memory Complexity:** \( O(r(d + k) + k) \)
</details>






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