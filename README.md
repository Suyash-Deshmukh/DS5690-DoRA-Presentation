# DoRA: Weight-<i>D</i>ecomposed L<i>o</i>w-<i>R</i>ank <i>A</i>daptation
### Authors: Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen

**Citation:**
Liu, Shih-Yang, Chien-Yi Wang, Hongxu Yin, et al. “DoRA: Weight-Decomposed Low-Rank Adaptation.” arXiv:2402.09353. Preprint, arXiv, July 9, 2024. https://doi.org/10.48550/arXiv.2402.09353. </br>
Published at the 41st International Conference on Machine Learning (ICML), 2024.

### Presented by: Suyash Deshmukh

---

## Paper Overview

[Five-minute overview providing context, stating the problem the paper is addressing, characterizing the approach, and giving a brief account of how the problem was addressed.]: #

Large pre-trained models like LLaMA and BERT achieve outstanding generalization, but full fine-tuning (FT) — updating every single parameter — is extremely expensive.

To make fine-tuning more efficient, researchers developed Parameter-Efficient Fine-Tuning (PEFT) methods, with LoRA (Low-Rank Adaptation) being one of the most popular.

LoRA inserts small trainable low-rank adapters into existing weight matrices, fine-tunes only those adapters, and merges them back after training.
It’s efficient, elegant, and adds no inference cost.

But - despite its success - LoRA consistently trails full fine-tuning in accuracy. 

Most prior work simply assumed this was due to LoRA’s limited number of trainable parameters, but never looked further into it. The authors of DoRA wanted to change that and investigate this assumption instead of blindly accepting it.

### Question 1:
You are a researcher at Nvidia and you want to get a better understanding of how FT and LoRA change the various matrices in your transformer model during training. How might you be able to <i>decompose</i> (take apart and take a closer look at) the changes in matrices?

<i>Hint: A Matrix is just another way to represent a vector. Scalars are just numbers (just magnitudes)... what is special about vectors?</i>
<details>
    <summary>
        Answer
    </summary>
  You can decompose vectors into a <u>magnitude</u> and a <u>direction</u>!
</details>

<br>

This base concept is what they exploited to understand what is happening under the hood and this is what led to the creation of DoRA!

So, what did they find when they looked into the decomposed matrix changes during training?

![FT vs LoRA slopes](images/ft_lora_slopes.jpg)

### Question 2:
After decomposing the changes in matrices during training, we can get a plot which shows change in direction ($\Delta D$) vs change in magnitude ($\Delta M$). Looking at comparison between FT and LoRA, what can we learn about the way they learn from training?

<i>Hint: There is a clear trend in both subfigures. Is this the same type of trend? What might each trend tell us about the relationship between $\Delta D$ and $\Delta M$?</i>
<details>
    <summary>
        Answer
    </summary>

  FT shows an inverse propotion between $\Delta D$ and $\Delta M$, while LoRA shows a direct propotion! This means that in LoRA bigger changes in direction are always accompanied by bigger changes in magnitude (and vice versa). This is because LoRA does not take advantage of decomposing the matrixes and is unable to fine-tune the direction <u>OR</u> magnitude- only both of them together.

  This is not the case for FT, which has a lot more freedom to learn minor changes and is capable of changing either the direction or the magnitude- leading to higher accuracy.
</details>

<br>

So, to get closer to FT in accuracy, we need the capability to fine-tune the magnitude and direction on their own! This revalation is what led to the creation of DoRA.

### DoRA functions as follows:
![Architecture of DoRA](images/dora_architecture.jpg)

1. DoRA decomposes each pre-trained weight W into a magnitude m and direction V.
2. It then fine-tunes both components separately:
    - The magnitude vector m is trained directly.
    - The directional component V is updated efficiently using LoRA’s low-rank matrices.
3. Recombines both parts after training - no extra inference cost, just like LoRA.

This lets DoRA adjust the scale and orientation of weights independently!

They find that DoRA outperforms LoRA on language, vision-language, and reasoning benchmarks; is able to get close to the accuracy of FT; can retain the efficiency of LoRA (only has minimal parameter increase); and can easily be integrated with other PEFT variants.

---

## Architecture Overview - Formal Algorithms
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

Initialize $A \leftarrow \text{Uniform Kaiming Distribution}$, $B \leftarrow 0$.</li>
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

Initialize $A \leftarrow \text{Uniform Kaiming Distribution}$, $B \leftarrow 0$, $\Delta m \leftarrow 0$.</li>
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

---

## Results!

Firstly, DoRA has a inversely propotional relationship between $\Delta D$ and $\Delta M$! Much more closely resembles FT than LoRA.
![FT vs LoRA vs DoRA slopes](images/ft_lora_dora_slopes.jpg)

### Commonsense reasoning:
| Model | Method | Trainable Params (%) | Avg. Accuracy |
|--------|---------|----------------------|------------------------|
| LLaMA2-7B | LoRA      | 0.83        | 77.6 |
| LLaMA2-7B | **DoRA*** | <i>0.43</i> | 80.5 |
| LLaMA2-7B | **DoRA**  | 0.84        | 79.7 |
| LLaMA3-8B | LoRA      | 0.70        | 80.8 |
| LLaMA3-8B | **DoRA*** | <i>0.35</i> | 85.0 |
| LLaMA3-8B | **DoRA**  | 0.71        | 85.2 |

### Visual question answering, Visual reasoning, and Image captioning:
Method | Trainable Params (%) | Avg. Accuracy |
|---------|----------------------|------------------------|
| FT        | 100         | 77.3 |
| LoRA      | 5.93        | 76.5 |
| **DoRA**  | 5.96        | 77.4 |

### Video question answering, and Video captioning:
Method | Trainable Params (%) | Avg. Accuracy |
|---------|----------------------|------------------------|
| FT        | 100         | 87.5 |
| LoRA      | 5.17        | 83.5 |
| **DoRA**  | 5.19        | 85.4 |

### Effect of Rank on Commonsense reasoning:
![Rank Sensitivity of LoRA vs DoRA](images/rank_sensitiviy.jpg)

### QDoRA: DoRA adaptation of QLoRA:

> [!NOTE]
> What is Quantized LoRA?
>
> PEFT significantly reduces training memory overhead, but we still need a considerable amount of GPU memory to initially load the model weights onto the GPUs. QLoRA suggests <i>quantizing</i> the base frozen model to 4-bit instead of 16-bit, and then fine-tuning with LoRA on top of that.

![QDoRA vs QLoRA accuracy](images/qdora.jpg)
---

## Critical Analysis
- A limitation of DoRA is that since it is designed to enhance LoRA’s performance to more closely resemble that of FT, in cases where LoRA performs better than / matches FT, the advantage of DoRA shrinks.

- The paper looks at text, images, and videos. However, they must further explore time-series domains such as audio to confirm the generality of directional fine-tuning.

- The comparisons made were to baseline PEFT models (LoRA/VeRA). In the rapidly evolving field that is ML, there are already new PEFT algorithms (such as PiSSA or LoFT). Work must be done to not only compare performances but to see if DoRA concepts might be able to be integrated with new PEFT algorithms for greater performance.

---

## Impact and Relevance
- It is able to get closer to the accuracy of FT while keeping the parameter and inference efficiency of LoRA.

- It is compatible and works as a drop-in replacement for LoRA and can be easily combined with other methods (E.g.: VeRA $\Rightarrow$ DVoRA)

- **Much like LoRA, DoRA lets researchers outside major AI labs fine-tune very large models using limited hardware.**
    - Researchers like us!

---

## Code Demonstration

This code snippet is from the GW-Whisper Project. These are the links to the [base GitHub page](https://github.com/vanderbilt-data-science/GW-Whisper/) as well as the [actual train.py](https://github.com/vanderbilt-data-science/GW-Whisper/blob/main/Glitch_classification/src/train.py) I took the code snippet from. If you wish to see and run this code for yourself please feel free to check this project out!

```py
whisper_model = WhisperModel.from_pretrained(f"openai/whisper-{args.encoder}").encoder.to(device)

module_names = [name for name, module in whisper_model.named_modules()]
patterns = ["layers.*.self_attn.q_proj", "layers.*.self_attn.k_proj", "layers.*.self_attn.v_proj", "layers.*.self_attn.o_proj"]

matched_modules = []
for pattern in patterns:
    matched_modules.extend(fnmatch.filter(module_names, pattern))

if args.method == 'LoRA':
    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=matched_modules)
    whisper_model_with_lora = get_peft_model(whisper_model, lora_config).to(device)

    for name, param in whisper_model_with_lora.named_parameters():
        param.requires_grad = 'lora' in name

elif args.method == 'DoRA':
    lora_config = LoraConfig(use_dora=True, r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=matched_modules)
    whisper_model_with_dora = get_peft_model(whisper_model, lora_config).to(device)

    for name, param in whisper_model_with_dora.named_parameters():
        param.requires_grad = 'lora' in name
```

Example of a model I ran with DoRA very recently (GitHub page for this project is not up yet as I am still working on the paper):

<img src="images/base_AST.png" alt="base AST" width="475"/>
<img src="images/trained_AST.png" alt="trained AST" width="500"/>

Getting this big of an improvement in $< 6$ hours of training on ACCRE. 
```java
trainable params: 470,016
all params: 86,657,280
trainable%: 0.5424%
```

---

## References + Resource Links
- [DoRA Paper (Liu et al., 2024)](https://arxiv.org/abs/2402.09353)
- [HuggingFace Page for DoRA](https://huggingface.co/papers/2402.09353)
- [Official DoRA Project Page](https://nbasyl.github.io/DoRA-project-page/)
- [DoRA GitHub Repository](https://github.com/NVlabs/DoRA)
- [Nvidia blogpost on DoRA](https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/)
- [LoRA Paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
- [VeRA Paper (Kopiczko et al., 2024)](https://arxiv.org/abs/2310.11454)
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)