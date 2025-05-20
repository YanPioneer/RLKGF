## RLKGF: Reinforcement Learning from Knowledge Graph Feedback Without Human Annotations

Reinforcement Learning from Human Feedback (RLHF) has been shown to effectively align large language models (LLMs) with human knowledge. However, the lack of human preference labels remains a significant bottleneck when applying RLHF to a downstream domain. Humans in RLHF play a critical role in injecting reasoning preferences into LLM, and we assume the reasoning process underlying human assessments may potentially be replaced by reasoning pathways derived from Knowledge Graphs (KGs). Inspired by this assumption, we propose Reinforcement Learning from Knowledge Graph Feedback (RLKGF), a novel method that leverages KG semantics and structure to derive RL rewards in the absence of manual annotations. Unlike Reinforcement Learning from AI Feedback (RLAIF), RLKGF directly integrates human priors encoded in KGs as the reward model, aligning LLM responses with expert knowledge without additional preference labeling or reward model training. RLKGF structures context-relevant facts into knowledge subgraphs and defines rewards by simulating information flow across semantic and logical connections between question and candidate response entities. Experiments on three public and one private medical dialogue dataset demonstrate that RLKGF significantly outperforms the competitive RLAIF in improving LLM diagnostic accuracy.

![img](https://github.com/YanPioneer/RLKGF/blob/main/image/Introduction_yan.png)

### Task Definition and Method
The disease diagnosis via Q\&A task requires the model to predict a disease $d$ in the answer $A$ based on a patient's symptom description $[s_1, s_2, ..., s_n]$ in the question $Q$.
Our focus is on using a medical knowledge graph (MKG) containing factual entities as a reward model to automatically assign feedback $R$ to model responses, i.e., RLKGF. 

The MKG $G = (V, E)$ is constructed based on the standard \cite{li2025quality}, where $V$ includes all symptom and disease entities in the dataset, and $E$ represents the relationships between these entities. In this task, we only consider the relationship < disease, causes, symptom >. After extracting the patient's information from the question, RLKGF first locates the relevant symptom entities in the MKG. It then identifies the disease entities connected to those symptoms and any other symptoms that these diseases may cause. Using the entities and their relationships, we construct the personalized diagnostic subgraph $g = (v, e)$, where $v$ represents the disease entities and related symptoms that may explain the patient's condition, and $e$ represents the corresponding triple relationships.

RLKGF evaluates the correctness of the model's response through path reasoning and semantic aggregation using graph-based random walk with restart (RWR) and GNNs. After acquiring feedback, RLKGF optimizes the model's policy using the proximal policy optimization (PPO) to align LLM responses with domain knowledge.

![img](https://github.com/YanPioneer/RLKGF/blob/main/image/main_figure_yan.png)


### Train

run xxxx/xxx_train.py

### Requirements

See requirement.txt

### Datasets

We utilize three public medical dialogue diagnosis datasets: MZ, DXY, and GMD. These datasets are derived from real-world medical dialogue diagnosis records.

We construct a new dataset, MED-D. MED-D is collected from offline electronic medical records (EMRs). These EMRs are sourced from cooperating hospitals and have been anonymized. We filter 14,277 EMRs and choose 20 diseases that could be diagnosed through Q\&A without additional tests. With the assistance of medical experts, we identify 351 associated symptoms. Subsequently, we extract symptom and disease entities from the selected EMRs using named entity recognition to construct Q\&A pairs. All extracted diseases and symptoms are manually aligned with the corresponding ICD-9 terms and reviewed by domain experts. We use the accuracy of disease prediction as the evaluation metric.

### Main Result
Table2 shows the performance of different LLMs trained using only RLAIF and RLKGF. \textbf{RLKGF Base} represents the accuracy achieved by directly selecting the response entity with the highest feedback score from the knowledge graph. From the experimental results, we observe the following.

\textbf{i. Advantages of RLKGF.} Our results demonstrate that RLKGF outperforms RLAIF by 5.67\%, 10.73\%, 8.38\%, and 1.21\% across four datasets, respectively. This indicates the feasibility and effectiveness of using KGs for feedback on model responses. It validates that leveraging KGs as reward models in the medical domain may be a more reliable approach than LLM-based preference labeling.

\textbf{ii. Small models are limited by instruction adherence.} Among different models, Qwen2.5-0.5b-instruct performs poorly, with only 32.39\% on the MZ dataset. We analyze its outputs before and after training and find that it has poor instruction adherence and fails to make correct predictions from the given diseases. Although training improves its instruction-following ability, knowledge injection remains suboptimal. We present the performance of models trained with supervised fine-tuning, where full-parameter SFT on Qwen2.5-0.5b-instruct achieves only 7.60\% accuracy. Preliminary analysis suggests that the MZ dataset's sparsity is insufficient to correct the initially learned model parameters. Additionally, smaller models may be more sensitive to loss design, and how to better inject knowledge into them requires further investigation.

\textbf{iii. Explore more effective KG feedback methods.} Furthermore, by comparing the prediction accuracy of models using only the KG, KG feedback-trained models, GPT-4o-mini predictions, RLAIF, and SFT, we find that although trained LLMs show some performance improvement, they still fall far short of the optimal target. Therefore, further exploration is needed on how to fully utilize factual knowledge and construct reasonable feedback to guide model training.

![img](https://github.com/YanPioneer/RLKGF/blob/main/image/image.png)


| Backbone              | Method                   | GMD        | DXY        | MZ         | MED-D      |
| --------------------- | ------------------------ | ---------- | ---------- | ---------- | ---------- |
| GPT-4o-mini           | Base                     | 0.646      | 0.4262     | 0.5289     | 0.5345     |
|                       | KG Prompt (Triple)       | 0.7569     | 0.7563     | 0.6275     | 0.6638     |
|                       | KG Prompt (Text)         | 0.6731     | 0.5772     | 0.5669     | \-         |
| Qwen2.5-3b-instruct   | Base                     | 0.6360     | 0.4531     | 0.3789     | 0.3553     |
|                       | FT                       | **0.7552** | 0.4951     | 0.4253     | 0.4823     |
|                       | LoRA                     | 0.7334     | 0.5038     | 0.4591     | **0.4843** |
|                       | KG Prompt (Triple)       | 0.7054     | 0.6495     | **0.6648** | 0.4671     |
|                       | KG Prompt (Text)         | 0.6452     | 0.5126     | 0.5768     | 0.4655     |
|                       | RLKGF                    | 0.7113     | **0.7314** | 0.6268     | 0.3800     |
|                       | RLKGF+KG Prompt (Triple) | 0.7490     | \-         | \-         | \-         |
| qwen2.5-1.5b-instruct | Base                     | 0.4840     | 0.2359     | 0.1845     | 0.1982     |
|                       | FT                       | 0.7066     | 0.438      | 0.4035     | **0.3863** |
|                       | LoRA                     | 0.7041     | 0.3543     | 0.3929     | 0.3543     |
|                       | KG Prompt (Triple)       | 0.6397     | 0.468      | 0.4352     | 0.3825     |
|                       | KG Prompt (Text)         | 0.6188     | 0.5757     | 0.4803     | 0.3575     |
|                       | RLKGF                    | 0.6109     | **0.5890** | **0.5070** | 0.3025     |
|                       | RLKGF+KG Prompt (Triple) | **0.7113** | \-         | \-         | \-         |
| qwen2.5-0.5b-instruct | Base                     | 0.2469     | 0.0981     | 0.0042     | 0.1273     |
|                       | FT                       | **0.4920** | **0.3388** | 0.0760     | **0.2510** |
|                       | LoRA                     | 0.4209     | 0.1252     | 0.0240     | 0.1860     |
|                       | KG Prompt (Triple)       | 0.4490     | 0.1515     | 0.0556     | 0.1525     |
|                       | KG Prompt (Text)         | 0.2870     | 0.1388     | 0.0373     | 0.1225     |
|                       | RLKGF                    | 0.3278     | 0.2654     | **0.3239** | 0.1475     |
|                       | RLKGF+KG Prompt (Triple) | 0.4519     | \-         | \-         | \-         |
| qwen1.5-4b-chat       | Base                     | 0.4038     | 0.4000     | 0.4176     | 0.1893     |
|                       | FT                       | 0.6866     | 0.6067     | 0.5260     | **0.3956** |
|                       | LoRA                     | 0.6485     | 0.6048     | 0.5556     | 0.3080     |
|                       | KG Prompt (Triple)       | 0.5540     | 0.4350     | 0.4754     | 0.1722     |
|                       | KG Prompt (Text)         | 0.4820     | 0.5184     | 0.3507     | 0.2022     |
|                       | RLKGF                    | 0.5914     | **0.6893** | **0.5986** | 0.2525     |
|                       | RLKGF+KG Prompt (Triple) | **0.6987** | \-         | \-         | \-         |
| qwen1.5-1.8b-chat     | Base                     | 0.3335     | 0.2291     | 0.0423     | 0.1342     |
|                       | FT                       | 0.5656     | 0.3320     | 0.2408     | **0.3233** |
|                       | LoRA                     | 0.5364     | 0.2864     | 0.1795     | 0.2600     |
|                       | KG Prompt (Triple)       | 0.2970     | 0.2786     | 0.0894     | 0.0392     |
|                       | KG Prompt (Text)         | 0.3326     | 0.2641     | 0.1188     | 0.1105     |
|                       | RLKGF                    | 0.4784     | **0.3366** | **0.3592** | 0.2050     |
|                       | RLKGF+KG Prompt (Triple) | **0.5690** | \-         | \-         | \-         |
| InternLM2.5-1.8b-chat | Base                     | 0.2092     | 0.3981     | 0.4507     | 0.1850     |
|                       | FT                       | **0.7573** | **0.7184** | **0.5985** | **0.4683** |
|                       | LoRA                     | 0.5828     | 0.5563     | 0.5859     | 0.3916     |
|                       | KG Prompt (Triple)       | 0.2594     | 0.4757     | 0.4648     | \-         |
|                       | KG Prompt (Text)         | 0.3180     | 0.3204     | 0.3380     | 0.1800     |
|                       | RLKGF                    | 0.5356     | 0.4757     | 0.5704     | 0.2025     |
|                       | RLKGF+KG Prompt (Triple) | 0.6192     | \-         | \-         | \-         |
| InternLM2-1.8b-chat   | Base                     | 0.3305     | 0.2718     | 0.2042     | 0.1667     |
|                       | FT                       | **0.7280** | **0.7766** | **0.6760** | **0.4836** |
|                       | LoRA                     | 0.7012     | 0.7116     | 0.6394     | 0.4080     |
|                       | KG Prompt (Triple)       | 0.2971     | 0.3883     | 0.0634     | \-         |
|                       | KG Prompt (Text)         | 0.3556     | 0.3204     | 0.0775     | 0.0875     |
|                       | RLKGF                    | 0.4686     | 0.4272     | 0.5282     | 0.2300     |
|                       | RLKGF+KG Prompt (Triple) | 0.4979     | \-         | \-         | \-         |

Table 9.  Comparison of Different Knowledge Injection Methods.

### RLKGF demonstrates better generalization capability compared to SFT.
We conduct an experiment to compare the generalization ability of RLKGF and SFT. Specifically, we use Qwen2.5-3B-Instruct as the base model, train it on the GMD dataset using both RLKGF and SFT, and evaluate the models on the DXY dataset. As shown in Table 4, the model trained with RLKGF demonstrates clear generalization to a different dataset, while the SFT-trained model even underperforms the untrained baseline, which highlights the superior generalization capability of RLKGF.
| Backbone            | Method       | DXY    |
| ------------------- | ------------ | ------ |
| Qwen2.5-3b-Instruct | Base         | 0.4531 |
|                     | RLKGF on GMD | 0.6117 |
|                     | SFT on GMD   | 0.3680 |

Table 4. Generalization comparison between RLKGF and SFT.


