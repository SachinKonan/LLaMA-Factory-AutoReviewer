# H2: Generated vs True Abstracts — Similarity Report

Total samples: 2019
Mean similarity: 0.7165
Median similarity: 0.7293
Std: 0.0888
Min: 0.2450, Max: 0.8984
>0.8: 323 (16.0%)
<0.5: 42 (2.1%)

---

## Top 20 Most Similar (Potential Contamination)

### Rank 1 — Similarity: 0.8984
**Title**: Online Coreset Selection for Rehearsal-based Continual Learning
**Year**: 2022 | **Decision**: Accept | **Submission**: f9D-5WNG4Nv
**Citation percentile**: 0.96 | **Ratings**: [5, 8, 6, 6]

**True Abstract**:
A dataset is a shred of crucial evidence to describe a task. However, each data point in the dataset does not have the same potential, as some of the data points can be more representative or informative than others. This unequal importance among the data points may have a large impact in rehearsal-based continual learning, where we store a subset of the training examples (coreset) to be replayed later to alleviate catastrophic forgetting. In continual learning, the quality of the samples stored in the coreset directly affects the model's effectiveness and efficiency. The coreset selection problem becomes even more important under realistic settings, such as imbalanced continual learning or noisy data scenarios. To tackle this problem, we propose Online Coreset Selection (OCS), a simple yet effective method that selects the most representative and informative coreset at each iteration and trains them in an online manner. Our proposed method maximizes the model's adaptation to a target dataset while selecting high-affinity samples to past tasks, which directly inhibits catastrophic forgetting. We validate the effectiveness of our coreset selection mechanism over various standard, imbalanced, and noisy datasets against strong continual learning baselines, demonstrating that it improves task adaptation and prevents catastrophic forgetting in a sample-efficient manner.

**Generated Abstract**:
Continual learning in machine learning settings often suffers from catastrophic forgetting, where models struggle to retain knowledge from previously learned tasks when exposed to new ones. To address this issue, rehearsal-based methods store a subset of the most relevant training examples from past tasks to periodically retrain the model, mitigating forgetting and improving long-term performance. However, storing the entire dataset can be computationally expensive and memory-intensive. This paper introduces an online coreset selection algorithm designed to efficiently manage the rehearsal set by dynamically selecting a small, representative subset (coreset) of examples from each new task. Our approach aims to minimize the size of the rehearsal set while preserving the model's performance on past tasks. We evaluate our method on several continual learning benchmarks and demonstrate significant improvements in computational efficiency without compromising on learning outcomes.

---

### Rank 2 — Similarity: 0.8964
**Title**: Progressive Prompts: Continual Learning for Language Models
**Year**: 2023 | **Decision**: Accept | **Submission**: UJTgQBc91_
**Citation percentile**: 0.94 | **Ratings**: [6, 8, 6, 8]

**True Abstract**:
We introduce Progressive Prompts – a simple and efficient approach for continual learning in language models. Our method allows forward transfer and resists catastrophic forgetting, without relying on data replay or a large number of task-specific parameters. Progressive Prompts learns a new soft prompt for each task and sequentially concatenates it with the previously learned prompts, while keeping the base model frozen. Experiments on standard continual learning benchmarks show that our approach outperforms state-of-the-art methods, with an improvement >20% in average test accuracy over the previous best-preforming method on T5 model. We also explore a more challenging continual learning setup with longer sequences of tasks and show that Progressive Prompts significantly outperforms prior methods.

**Generated Abstract**:
We introduce Progressive Prompts, a novel continual learning approach designed for language models to efficiently acquire new knowledge without forgetting previously learned information. This method employs a hierarchical prompt structure that allows the model to incrementally build upon its existing knowledge base. Through a series of carefully crafted prompts, Progressive Prompts enable language models to adapt to new tasks while maintaining performance on older tasks, addressing the challenge of catastrophic forgetting in continual learning settings. We evaluate the proposed technique on a variety of natural language processing benchmarks and demonstrate its effectiveness in both zero-shot and few-shot learning scenarios, showing significant improvements in task adaptation and long-term retention.

---

### Rank 3 — Similarity: 0.8912
**Title**: EmpathyRobot: A Dataset and Benchmark for Empathetic Task Planning of Robotic Agent
**Year**: 2025 | **Decision**: Reject | **Submission**: F6h0v1CTpC
**Citation percentile**: 0.29 | **Ratings**: [5, 3, 8, 8]

**True Abstract**:
Empathy is a fundamental instinct and essential need for humans, as they both demonstrate empathetic actions toward others and receive empathetic support. As robots become increasingly integrated into daily life, it is essential to explore whether they can provide human-like empathetic support. Although existing emotion agents have explored how to understand humans' empathetic needs, they lack to further enable robots to generate empathy-oriented task planning, neglecting the evaluation of empathetic behaviors. To address this gap, we introduce \textbf{EmpathyRobot}, the first dataset specifically designed to benchmark and enhance the empathetic actions of agents across diverse scenarios. This dataset contains 10,000 samples based on human feedback, encompassing information from various modalities and corresponding empathetic task planning sequences, including navigation and manipulation. Agents are required to perform actions based on their understanding of both the visual scene and human emotions. To systematically evaluate the performance of existing agents on the EmpathyRobot dataset, we conduct comprehensive experiments to test the most capable models. Our findings reveal that generating accurate empathetic actions remains a significant challenge. Meanwhile, we finetune an \ac{llm} on our benchmark, demonstrating that it can effectively be used to enhance the empathetic behavior of robot agents. By establishing a standard benchmark for evaluating empathetic actions, we aim to drive advancements in the study and pursue of empathetic behaviors in robot agents. We will release our code and dataset.

**Generated Abstract**:
This paper introduces EmpathyRobot, a comprehensive dataset and benchmark designed to evaluate the ability of robotic agents to plan empathetic actions in response to human emotional states. The dataset consists of scenarios where a robotic agent must decide on appropriate actions to take in order to demonstrate empathy towards humans experiencing various emotional conditions. We present a methodology for collecting and annotating these scenarios, and we propose a set of evaluation metrics to assess the performance of different task planning algorithms in generating empathetic behaviors. EmpathyRobot is intended to serve as a valuable resource for advancing the field of human-robot interaction by providing a standardized platform for comparing and improving empathetic task planning capabilities in robotic agents.

---

### Rank 4 — Similarity: 0.8892
**Title**: Differentially Private Adaptive Optimization with Delayed Preconditioners
**Year**: 2023 | **Decision**: Accept | **Submission**: j1zQGmQQOX1
**Citation percentile**: 0.55 | **Ratings**: [3, 8, 6, 6]

**True Abstract**:
Privacy costs may negate the benefits of using adaptive optimizers in differentially private model training. Prior works typically address this issue by using auxiliary information (e.g., public data) to boost the effectiveness of adaptive optimization. In this work, we explore techniques to estimate and efficiently adapt to gradient geometry in private adaptive optimization without auxiliary data. Motivated by the observation that adaptive methods can tolerate stale preconditioners, we propose differentially private adaptive training with delayed preconditioners (DP^2), a simple method that constructs delayed but less noisy preconditioners to better realize the benefits of adaptivity. Theoretically, we provide convergence guarantees for our method for both convex and non-convex problems, and analyze trade-offs between delay and privacy noise reduction. Empirically, we explore DP^2 across several real-world datasets, demonstrating that it can improve convergence speed by as much as 4× relative to non-adaptive baselines and match the performance of state-of-the-art optimization methods that require auxiliary data.

**Generated Abstract**:
We propose a differentially private adaptive optimization algorithm that incorporates delayed preconditioners to improve the utility of the model while preserving privacy. Our method addresses the challenge of maintaining model accuracy and convergence rates in the presence of privacy constraints, by leveraging the benefits of adaptive optimization techniques and the advantages of delayed preconditioning. We demonstrate that our approach achieves competitive performance on various machine learning benchmarks compared to non-private adaptive optimization methods, while ensuring strong privacy guarantees through differential privacy mechanisms.

---

### Rank 5 — Similarity: 0.8889
**Title**: Efficient Sampling for Generative Adversarial Networks with Reparameterized Markov Chains
**Year**: 2021 | **Decision**: Reject | **Submission**: c7rtqjVaWiE
**Citation percentile**: 0.09 | **Ratings**: [7, 5, 5, 8]

**True Abstract**:
Recently, sampling methods have been successfully applied to enhance the sample quality of Generative Adversarial Networks (GANs). However, in practice, they typically have poor sample efficiency because of the independent proposal sampling from the generator. In this work, we propose REP-GAN, a novel sampling method that allows general dependent proposals by REParameterizing the Markov chains into the latent space of the generator. Theoretically, we show that our reparameterized proposal admits a closed-form Metropolis-Hastings acceptance ratio. Empirically, extensive experiments on synthetic and real datasets demonstrate that our REP-GAN largely improves the sample efficiency and obtains better sample quality simultaneously.

**Generated Abstract**:
We propose a novel sampling method for Generative Adversarial Networks (GANs) that leverages reparameterized Markov chains to improve both the efficiency and quality of generated samples. By reparameterizing the Markov chains, we enable more effective exploration of the data distribution, leading to faster convergence and better sample diversity. Our method reduces the computational overhead associated with traditional sampling techniques, making it particularly suitable for high-dimensional and complex datasets. Experimental results demonstrate significant improvements in sample quality and generation speed across various GAN architectures and dataset types, outperforming existing sampling approaches in terms of both efficiency and effectiveness.

---

### Rank 6 — Similarity: 0.8877
**Title**: SELFIES-TED : A Robust Transformer Model for Molecular Representation using SELFIES
**Year**: 2025 | **Decision**: Reject | **Submission**: uPj9oBH80V
**Citation percentile**: 0.29 | **Ratings**: [6, 1, 6, 1]

**True Abstract**:
Large-scale molecular representation methods have revolutionized applications in material science, such as drug discovery, chemical modeling, and material design. With the rise of transformers, models now learn representations directly from molecular structures. In this paper, we introduce SELFIES-TED, a transformer-based model designed for molecular representation using SELFIES, a more robust, unambiguous method for encoding molecules compared to traditional SMILES strings. By leveraging the robustness of SELFIES and the power of the transformer encoder-decoder architecture, SELFIES-TED effectively captures the intricate relationships between molecular structures and their properties. Having pretrained with 1 billion molecule samples, our model demonstrates improved performance on molecular property prediction tasks across various benchmarks, showcasing its generalizability and robustness. Additionally, we explore the latent space of SELFIES-TED, revealing valuable insights that enhance its capabilities in both molecule property prediction and molecule generation tasks, opening new avenues for innovation in molecular design.

**Generated Abstract**:
We introduce SELFIES-TED, a novel approach for representing molecular structures using transformers. SELFIES (SMILES-like Expressions for Flexible Encoding of Smiles) is a flexible and robust encoding scheme for molecules, which we leverage to convert molecular structures into text-like sequences. These sequences are then fed into a transformer model for efficient and accurate molecular representation. We evaluate SELFIES-TED on various molecular property prediction and molecular generation tasks, demonstrating its superior performance and robustness compared to existing methods, particularly in handling noisy and ambiguous molecular inputs.

---

### Rank 7 — Similarity: 0.8852
**Title**: Policy-Driven Attack: Learning to Query for Hard-label Black-box Adversarial Examples
**Year**: 2021 | **Decision**: Accept | **Submission**: pzpytjk3Xb2
**Citation percentile**: 0.53 | **Ratings**: [7, 6, 7, 7]

**True Abstract**:
To craft black-box adversarial examples, adversaries need to query the victim model and take proper advantage of its feedback. Existing black-box attacks generally suffer from high query complexity, especially when only the top-1 decision (i.e., the hard-label prediction) of the victim model is available. In this paper, we propose a novel hard-label black-box attack named Policy-Driven Attack, to reduce the query complexity. Our core idea is to learn promising search directions of the adversarial examples using a well-designed policy network in a novel reinforcement learning formulation, in which the queries become more sensible. Experimental results demonstrate that our method can significantly reduce the query complexity in comparison with existing state-of-the-art hard-label black-box attacks on various image classification benchmark datasets.

**Generated Abstract**:
This paper introduces a novel approach called Policy-Driven Attack (PDA) for generating hard-label black-box adversarial examples. PDA leverages a policy gradient method to iteratively query a target model and refine adversarial examples, aiming to maximize the attack success rate while minimizing the number of queries. By learning an optimal querying policy, PDA efficiently discovers adversarial examples that can deceive the target model with fewer interactions, outperforming existing black-box attack techniques in terms of query efficiency and attack effectiveness across various machine learning models.

---

### Rank 8 — Similarity: 0.8851
**Title**: On the Impact of Adversarially Robust Models on Algorithmic Recourse
**Year**: 2023 | **Decision**: Reject | **Submission**: BGId14emsBj
**Citation percentile**: 0.32 | **Ratings**: [3, 3, 6, 3]

**True Abstract**:
The widespread deployment of machine learning models in various high-stakes settings has underscored the need for ensuring that individuals who are adversely impacted by model predictions are provided with a means for recourse. To this end, several algorithms have been proposed in recent literature to generate recourses. Recent research has also demonstrated that the recourses generated by these algorithms often correspond to adversarial examples. This key finding emphasizes the need for a deeper understanding of the impact of adversarially robust models (which are designed to guard against adversarial examples) on algorithmic recourse. In this work, we make one of the first attempts at studying the impact of adversarially robust models on algorithmic recourse. We theoretically and empirically analyze the cost (ease of implementation) and validity (probability of obtaining a positive model prediction) of the recourses output by state-of-the-art algorithms when the underlying models are adversarially robust. More specifically, we construct theoretical bounds on the differences between the cost and the validity of the recourses generated by various state-of-the-art algorithms when the underlying models are adversarially robust vs. non-robust. We also carry out extensive empirical analysis with multiple real-world datasets to not only validate our theoretical results, but also analyze the impact of varying degrees of model robustness on the cost and validity of the resulting recourses. Our theoretical and empirical analyses demonstrate that adversarially robust models significantly increase the cost and reduce the validity of the resulting recourses, thereby shedding light on the inherent trade-offs between achieving adversarial robustness in predictive models and providing easy-to-implement and reliable algorithmic recourse.

**Generated Abstract**:
This paper explores the implications of adversarially robust machine learning models on the ability to provide algorithmic recourse for decision-making processes. We analyze how robustness against adversarial attacks influences the interpretability and stability of model predictions, and subsequently, the feasibility of offering actionable steps to individuals affected by incorrect or unfavorable outcomes. Our empirical evaluations demonstrate that while adversarial robustness improves model resilience, it often comes at the cost of reduced explainability and increased computational complexity, which can hinder the provision of effective recourse mechanisms. We propose a framework for evaluating the trade-offs between robustness and recourse, and discuss potential strategies to mitigate these challenges.

---

### Rank 9 — Similarity: 0.8845
**Title**: Correcting Data Distribution Mismatch in Offline Meta-Reinforcement Learning with Few-Shot Online Adaptation
**Year**: 2023 | **Decision**: Reject | **Submission**: Dk7tsv9fkF
**Citation percentile**: 0.11 | **Ratings**: [5, 5, 6, 5]

**True Abstract**:
Offline meta-reinforcement learning (offline meta-RL) extracts knowledge from a given dataset of multiple tasks and achieves fast adaptation to new tasks. Recent offline meta-RL methods typically use task-dependent behavior policies (e.g., training RL agents on each individual task) to collect a multi-task dataset and learn an offline meta-policy. However, these methods always require extra information for fast adaptation, such as offline context for testing tasks or oracle reward functions. Offline meta-RL with few-shot online adaptation remains an open problem. In this paper, we first formally characterize a unique challenge under this setting: data distribution mismatch between offline training and online adaptation. This distribution mismatch may lead to unreliable offline policy evaluation and the regular adaptation methods of online meta-RL will suffer. To address this challenge, we introduce a novel mechanism of data distribution correction, which ensures the consistency between offline and online evaluation by filtering out out-of-distribution episodes in online adaptation. As few-shot out-of-distribution episodes usually have lower returns, we propose a Greedy Context-based data distribution Correction approach, called GCC, which greedily infers how to solve new tasks. GCC diversely samples “task hypotheses” from the current posterior belief and selects a greedy hypothesis with the highest return to update the task belief. Our method is the first to provide an effective online adaptation without additional information, and can be combined with off-the-shelf context-based offline meta-training algorithms. Empirical experiments show that GCC achieves state-of-the-art performance on the Meta-World ML1 benchmark compared to baselines with/without offline adaptation.

**Generated Abstract**:
This paper addresses the challenge of data distribution mismatch in offline meta-reinforcement learning (meta-RL) by proposing a novel framework that incorporates few-shot online adaptation. The approach aims to improve the generalization ability of meta-RL algorithms when applied to new tasks with different data distributions than those seen during training. By leveraging offline datasets from diverse tasks and employing online adaptation techniques, the method enables the model to quickly adjust its learned policies to match the characteristics of the new task, thereby enhancing its performance and robustness in real-world applications. Experimental results demonstrate significant improvements in adaptation speed and effectiveness compared to existing methods, showcasing the potential of this framework to advance the field of meta-RL.

---

### Rank 10 — Similarity: 0.8836
**Title**: Anti-Correlated Noise in Epoch-Based Stochastic Gradient Descent: Implications for Weight Variances
**Year**: 2025 | **Decision**: Reject | **Submission**: FV6rPMwmuG
**Citation percentile**: 0.29 | **Ratings**: [3, 6, 6, 5]

**True Abstract**:
Stochastic Gradient Descent (SGD) has become a cornerstone of neural network optimization due to its computational efficiency and generalization capabilities. However, the noise introduced by SGD is often assumed to be uncorrelated over time, despite the common practice of epoch-based training where data is sampled without replacement. In this work, we challenge this assumption and investigate the effects of epoch-based noise correlations on the stationary distribution of discrete-time SGD with momentum. Our main contributions are twofold: First, we calculate the exact autocorrelation of the noise during epoch-based training under the assumption that the noise is independent of small fluctuations in the weight vector, revealing that SGD noise is inherently anti-correlated over time. Second, we explore the influence of these anti-correlations on the variance of weight fluctuations. We find that for directions with curvature of the loss greater than a hyperparameter-dependent crossover value, the conventional results for uncorrelated noise are recovered. However, for relatively flat directions, the weight variance is significantly reduced, leading to a considerable decrease in loss fluctuations compared to the constant weight variance assumption. Furthermore, we demonstrate that training with these anti-correlations enhances test performance, suggesting that the inherent noise structure induced by epoch-based training plays a crucial role in finding flatter minima that generalize better.

**Generated Abstract**:
We investigate the phenomenon of anti-correlated noise in Epoch-Based Stochastic Gradient Descent (SGD) and its implications on weight variances. By analyzing the behavior of SGD during training, we find that the noise in consecutive updates is often anti-correlated, leading to a unique pattern in the evolution of weight variances. This anti-correlation can significantly affect the convergence dynamics and generalization performance of deep learning models. Our theoretical analysis and empirical studies demonstrate that accounting for anti-correlated noise provides a more accurate understanding of the optimization process and can lead to improved training strategies.

---

### Rank 11 — Similarity: 0.8834
**Title**: MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models
**Year**: 2025 | **Decision**: Accept | **Submission**: Usklli4gMc
**Citation percentile**: 0.91 | **Ratings**: [5, 6, 5, 6, 6]

**True Abstract**:
Existing multimodal retrieval benchmarks primarily focus on evaluating whether models can retrieve and utilize external textual knowledge for question answering. However, there are scenarios where retrieving visual information is either more beneficial or easier to access than textual data. In this paper, we introduce a multimodal retrieval-augmented generation benchmark, MRAG-Bench, in which we systematically identify and categorize scenarios where visually augmented knowledge is better than textual knowledge, for instance, more images from varying viewpoints. MRAG-Bench consists of 16,130 images and 1,353 human-annotated multiple-choice questions across 9 distinct scenarios. With MRAG-Bench, we conduct an evaluation of 10 open-source and 4 proprietary large vision-language models (LVLMs). Our results show that all LVLMs exhibit greater improvements when augmented with images compared to textual knowledge, confirming that MRAG-Bench is vision-centric. Additionally, we conduct extensive analysis with MRAG-Bench, which offers valuable insights into retrieval-augmented LVLMs. Notably, the top-performing model, GPT-4o, faces challenges in effectively leveraging retrieved knowledge, achieving only a 5.82\% improvement with ground-truth information, in contrast to a 33.16\% improvement observed in human participants. These findings highlight the importance of MRAG-Bench in encouraging the community to enhance LVLMs' ability to utilize retrieved visual knowledge more effectively.


![Previous Benchmarks Retrieve textual-intensive knowledge](images/f9c26884e651ffa539da6cfa34625827920f047afb757cec1f5610c7b5123028.jpg)


![We focus on scenarios where 1.Retrieving correct textual knowledge is hard 2.Retrieved correct textual knowledge is not useful](images/9b547489d40ccaaeb8c5b80a1004b09fecd26f34a123d1f2c140b0dc8994bca4.jpg)


![](images/b79e07ff339619a8dfe0d89b0509fb030c05756981326a894375399a568b1b43.jpg)

**Generated Abstract**:
We introduce MRAG-Bench, a novel benchmark designed to evaluate the performance of retrieval-augmented multimodal models in vision-centric tasks. This benchmark focuses on scenarios where visual information plays a crucial role in the model's ability to generate accurate and contextually rich responses. By integrating large-scale multimodal datasets and carefully crafted evaluation metrics, MRAG-Bench aims to provide a comprehensive and fair assessment of the capabilities of these models in handling complex visual and textual information. Our evaluation framework includes both quantitative and qualitative measures to capture the nuanced performance of retrieval-augmented models across various aspects of multimodal understanding and generation. We demonstrate the utility of MRAG-Bench by applying it to a range of state-of-the-art models and identifying areas for improvement in their visual reasoning and retrieval mechanisms.

---

### Rank 12 — Similarity: 0.8815
**Title**: NTK-DFL: Enhancing Decentralized Federated Learning in Heterogeneous Settings via Neural Tangent Kernel
**Year**: 2025 | **Decision**: Reject | **Submission**: TwZBQKgwdW
**Citation percentile**: 0.29 | **Ratings**: [5, 6, 5, 5]

**True Abstract**:
Decentralized federated learning (DFL) is a collaborative machine learning framework for training a model across participants without a central server or raw data exchange. DFL faces challenges due to statistical heterogeneity, as participants often possess different data distributions reflecting local environments and user behaviors. Recent work has shown that the neural tangent kernel (NTK) approach, when applied to federated learning in a centralized framework, can lead to improved performance. The NTK-based update mechanism is more expressive than typical gradient descent methods, enabling more efficient convergence and better handling of data heterogeneity. We propose an approach leveraging the NTK to train client models in the decentralized setting, while introducing a synergy between NTK-based evolution and model averaging. This synergy exploits inter-model variance and improves both accuracy and convergence in heterogeneous settings. Our model averaging technique significantly enhances performance, boosting accuracy by at least 10% compared to the mean local model accuracy. Empirical results demonstrate that our approach consistently achieves higher accuracy than baselines in highly heterogeneous settings, where other approaches often underperform. Additionally, it reaches target performance in 4.6 times fewer communication rounds. We validate our approach across multiple datasets, network topologies, and heterogeneity settings to ensure robustness and generalizability. The source code will be available as a link on the discussion forum once it is open.

**Generated Abstract**:
In this paper, we introduce NTK-DFL, a novel approach to enhance decentralized federated learning (DFL) in heterogeneous settings. By leveraging the properties of the Neural Tangent Kernel (NTK), we aim to mitigate the challenges posed by data heterogeneity and communication constraints in DFL. Our method stabilizes the learning process, accelerates convergence, and improves model performance across diverse client datasets. We theoretically analyze the behavior of NTK in the context of DFL and empirically demonstrate its effectiveness through extensive experiments on various real-world datasets. The results show that NTK-DFL achieves superior accuracy compared to existing DFL algorithms, particularly in scenarios with significant data heterogeneity.

---

### Rank 13 — Similarity: 0.8790
**Title**: LVLM-COUNT: Enhancing the Counting Ability of Large Vision-Language Models
**Year**: 2025 | **Decision**: Reject | **Submission**: GsCMKwyfWm
**Citation percentile**: 0.64 | **Ratings**: [5, 6, 3, 6]

**True Abstract**:
Counting is a fundamental skill for various visual tasks in real-life applications, requiring both object recognition and robust counting capabilities. Despite their advanced visual perception, large vision-language models (LVLMs) struggle with counting tasks, especially when the number of objects exceeds those commonly encountered during training. We enhance LVLMs’ counting abilities using a divide-and conquer approach, breaking counting problems into sub-counting tasks. Unlike prior methods, which do not generalize well to counting datasets on which they have not been trained, our method performs well on new datasets without any additional training or fine-tuning. We demonstrate that our approach enhances counting capabilities across various datasets and benchmarks.

**Generated Abstract**:
This paper introduces LVLM-COUNT, a novel method designed to improve the counting ability of large vision-language models. By integrating a specialized counting module into the existing architecture, LVLM-COUNT addresses the limitations of current models in accurately performing counting tasks on images, especially when dealing with complex scenes and large object numbers. Our approach leverages advanced multimodal representations and task-specific training strategies to enhance the model's understanding and manipulation of numerical information within visual contexts. Extensive experiments on standard benchmarks demonstrate that LVLM-COUNT significantly outperforms existing models in various counting scenarios, making it a valuable addition to the field of vision-language understanding.

---

### Rank 14 — Similarity: 0.8783
**Title**: Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization
**Year**: 2025 | **Decision**: Accept | **Submission**: omrLHFzC37
**Citation percentile**: 0.86 | **Ratings**: [5, 6, 6, 8]

**True Abstract**:
Federated Learning (FL) offers a promising framework for collaborative and privacy-preserving machine learning across distributed data sources. However, the substantial communication costs associated with FL significantly challenge its efficiency. Specifically, in each communication round, the communication costs scale linearly with the model's dimension, which presents a formidable obstacle, especially in large model scenarios. Despite various communication-efficient strategies, the intrinsic dimension-dependent communication cost remains a major bottleneck for current FL implementations. This paper proposes a novel dimension-free communication algorithm - DeComFL, which leverages the zeroth-order optimization techniques and reduces the communication cost from $\mathcal{O}(d)$ to $\mathcal{O}(1)$ by transmitting only a constant number of scalar values between clients and the server in each round, regardless of the dimension $d$ of the model parameters. Theoretically, in non-convex functions, we prove that our algorithm achieves state-of-the-art rates, which show a linear speedup of the number of clients and local steps under standard assumptions. With additional low effective rank assumption, we can further show that the convergence rate is independent of the model dimension $d$ as well. Empirical evaluations, encompassing both classic deep learning training and large language model fine-tuning, demonstrate significant reductions in communication overhead. Notably, DeComFL achieves this by transmitting only around 1MB of data in total between the server and a client to fine-tune a model with billions of parameters.

**Generated Abstract**:
Federated learning (FL) enables multiple devices or entities to collaboratively train a model while keeping their data local, which is crucial for privacy preservation. However, FL faces significant challenges, particularly in high-dimensional settings where the communication overhead between the central server and the clients becomes prohibitive. This paper proposes a novel zeroth-order optimization (ZOO) framework to address these challenges by reducing the communication cost to be independent of the model dimension. The method leverages random perturbations and unbiased gradient estimators to approximate the gradient at the central server without requiring explicit model updates from the clients. We demonstrate that our approach achieves dimension-free communication complexity and shows comparable or superior performance to first-order methods in various federated learning scenarios, including non-convex optimization problems. Our experiments on real-world datasets confirm the effectiveness of the proposed framework in reducing communication costs while maintaining model accuracy.

---

### Rank 15 — Similarity: 0.8782
**Title**: ECD: A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Crystalline Inorganic Materials
**Year**: 2025 | **Decision**: Accept | **Submission**: SBCMNc3Mq3
**Citation percentile**: 0.29 | **Ratings**: [6, 6, 8, 6]

**True Abstract**:
Supervised machine learning techniques are increasingly being adopted to speed up electronic structure predictions, serving as alternatives to first-principles methods like Density Functional Theory (DFT). Although current DFT datasets mainly emphasize chemical properties and atomic forces, the precise prediction of electronic charge density is essential for accurately determining a system's total energy and ground state properties. In this study, we introduce a novel electronic charge density dataset named ECD, which encompasses 140,646 stable crystal geometries with medium-precision Perdew–Burke–Ernzerhof (PBE) functional data. Within this dataset, a subset of 7,147 geometries includes high-precision electronic charge density data calculated using the Heyd–Scuseria–Ernzerhof (HSE) functional in DFT. By designing various benchmark tasks for crystalline materials and emphasizing training with large-scale PBE data while fine-tuning with a smaller subset of high-precision HSE data, we demonstrate the efficacy of current machine learning models in predicting electronic charge densities. The ECD dataset and baseline models are open-sourced to support community efforts in developing new methodologies and accelerating materials design and applications.

**Generated Abstract**:
This paper introduces ECD, a machine learning benchmark designed to predict enhanced-precision electronic charge density in crystalline inorganic materials. By leveraging advanced machine learning techniques, ECD aims to improve the accuracy and efficiency of charge density calculations, which are crucial for understanding electronic structure and properties of materials at the atomic level. The benchmark includes a diverse dataset of materials with known charge densities, allowing for the evaluation and comparison of different machine learning models. Results demonstrate that ECD can significantly enhance the precision of charge density predictions, providing valuable insights for material science and related fields.

---

### Rank 16 — Similarity: 0.8776
**Title**: Dynamic Influence Tracker: Estimating Sample Influence in SGD-Trained Models across Arbitrary Time Windows
**Year**: 2025 | **Decision**: Reject | **Submission**: g1kSMVqaXg
**Citation percentile**: 0.29 | **Ratings**: [5, 6, 3, 8, 3]

**True Abstract**:
Understanding how training samples affect models improves model interpretability, optimization strategies, and anomaly detection. However, existing methods for estimating sample influence provide only static assessments, rely on restrictive assumptions, and require high computational costs. We propose Dynamic Influence Tracker (DIT), a novel method to estimate time-varying sample influence in models trained with Stochastic Gradient Descent (SGD). DIT enables fine-grained analysis of sample influence within arbitrary time windows during training through a two-phase algorithm. The training phase efficiently captures and stores necessary information about the SGD trajectory, while the inference phase computes the influence of samples on the model within a specified time window. We provide a theoretical error bound for our estimator without assuming convexity, showing its reliability across various learning scenarios. Our experimental results reveal the evolution of sample influence throughout the training process, enhancing understanding of learning dynamics. We show DIT's effectiveness in improving model performance through anomalous sample detection and its potential for advancing curriculum learning.

**Generated Abstract**:
We introduce Dynamic Influence Tracker (DIT), a novel method for estimating the influence of training samples on the parameters of Stochastic Gradient Descent (SGD)-trained models at any arbitrary time window during training. DIT provides a principled way to track how the importance of samples changes over time, enabling a deeper understanding of the learning dynamics and the role of individual data points in the model's final parameters. This technique is particularly useful for identifying influential samples that drive model performance and for diagnosing issues such as overfitting or data imbalance. Through extensive experiments on various datasets and model architectures, we demonstrate that DIT can accurately estimate sample influence, offering insights that are critical for improving model robustness and generalization.

---

### Rank 17 — Similarity: 0.8774
**Title**: Personalized Neural Architecture Search for Federated Learning
**Year**: 2022 | **Decision**: Reject | **Submission**: WcZUevpX3H3
**Citation percentile**: 0.35 | **Ratings**: [3, 5, 5, 5]

**True Abstract**:
Federated Learning (FL) is a recently proposed learning paradigm for decentralized devices to collaboratively train a predictive model without exchanging private data. Existing FL frameworks, however, assume a one-size-fit-all model architecture to be collectively trained by local devices, which is determined prior to observing their data. Even with good engineering acumen, this often falls apart when local tasks are different and require diverging choices of architecture modelling to learn effectively. This motivates us to develop a novel personalized neural architecture search (NAS) algorithm for FL. Our algorithm, FedPNAS, learns a base architecture that can be structurally personalized for quick adaptation to each local task. We empirically show that FedPNAS significantly outperforms other NAS and FL benchmarks on several real-world datasets.

**Generated Abstract**:
Federated learning (FL) enables the training of machine learning models across multiple decentralized devices or servers holding local data samples, without exchanging them. However, the one-size-fits-all approach to neural architecture search (NAS) in FL often fails to optimize for the diverse and heterogeneous nature of client data. This paper proposes a personalized NAS framework for federated learning, which dynamically searches and adapts neural architectures to the specific characteristics of each client's data. Our approach significantly improves model performance and efficiency by leveraging local data statistics and communication constraints. Experimental results demonstrate that the personalized architectures outperform generic NAS solutions in terms of accuracy and resource utilization across various FL scenarios.

---

### Rank 18 — Similarity: 0.8768
**Title**: 3D Pre-training improves GNNs for Molecular Property Prediction
**Year**: 2022 | **Decision**: Reject | **Submission**: LNmNWds-q-J
**Citation percentile**: 0.11 | **Ratings**: [5, 3, 8, 6]

**True Abstract**:
Molecular property prediction is one of the fastest-growing applications of deep learning with critical real-world impacts. Including 3D molecular structure as input to learned models their performance for many molecular tasks. However, this information is infeasible to compute at the scale required by several real-world applications. We propose pre-training a model to reason about the geometry of molecules given only their 2D molecular graphs. Using methods from self-supervised learning, we maximize the mutual information between 3D summary vectors and the representations of a Graph Neural Network (GNN) such that they contain latent 3D information. During fine-tuning on molecules with unknown geometry, the GNN still generates implicit 3D information and can use it to improve downstream tasks. We show that 3D pre-training provides significant improvements for a wide range of properties, such as a 22% average MAE reduction on eight quantum mechanical properties. Moreover, the learned representations can be effectively transferred between datasets in different molecular spaces.

**Generated Abstract**:
This paper explores the effectiveness of 3D pre-training in enhancing the performance of Graph Neural Networks (GNNs) for molecular property prediction. By leveraging 3D structural information of molecules, we introduce a novel pre-training method that significantly improves the predictive accuracy of GNNs on various molecular datasets. Our approach involves learning embeddings that capture the spatial and chemical properties of molecules in a 3D context, which are then fine-tuned for specific prediction tasks. Experimental results demonstrate that 3D pre-trained GNNs outperform their 2D counterparts and other state-of-the-art methods, showcasing the potential of integrating 3D information into graph-based models for more robust and accurate molecular property predictions.

---

### Rank 19 — Similarity: 0.8768
**Title**: On Fast Adversarial Robustness Adaptation in Model-Agnostic Meta-Learning
**Year**: 2021 | **Decision**: Accept | **Submission**: o81ZyBCojoA
**Citation percentile**: 0.71 | **Ratings**: [6, 6, 6, 6]

**True Abstract**:
Model-agnostic meta-learning (MAML) has emerged as one of the most successful meta-learning techniques in few-shot learning. It enables us to learn a $\textit{meta-initialization}$ of model parameters (that we call $\textit{meta-model}$) to rapidly adapt to new tasks using a small amount of labeled training data. Despite the generalization power of the meta-model, it remains elusive that how $\textit{adversarial robustness}$ can be maintained by MAML in few-shot learning. In addition to generalization, robustness is also desired for a meta-model to defend adversarial examples (attacks). Toward promoting adversarial robustness in MAML, we first study $\textit{when}$ a robustness-promoting regularization should be incorporated, given the fact that MAML adopts a bi-level (fine-tuning vs. meta-update) learning procedure. We show that robustifying the meta-update stage is sufficient to make robustness adapted to the task-specific fine-tuning stage even if the latter uses a standard training protocol. We also make additional justification on the acquired robustness adaptation by peering into the interpretability of neurons' activation maps. Furthermore, we investigate $\textit{how}$ robust regularization can $\textit{efficiently}$ be designed in MAML. We propose a general but easily-optimized robustness-regularized meta-learning framework, which allows the use of unlabeled data augmentation, fast adversarial attack generation, and computationally-light fine-tuning. In particular, we for the first time show that the auxiliary contrastive learning task can enhance the adversarial robustness of MAML. Finally, extensive experiments are conducted to demonstrate the effectiveness of our proposed methods in robust few-shot learning.

**Generated Abstract**:
This paper addresses the challenge of enhancing adversarial robustness in model-agnostic meta-learning (MAML) frameworks. We propose a novel adaptation mechanism that accelerates the robustness acquisition process for models trained via MAML, particularly in few-shot learning scenarios. Our approach involves modifying the standard MAML algorithm to incorporate adversarial training steps during the inner-loop adaptation, thereby enabling faster and more effective adaptation to adversarial examples. Through extensive experiments on various few-shot learning datasets, we demonstrate that our method significantly improves the robustness of MAML-trained models with minimal overhead, outperforming existing robustness adaptation techniques in terms of both speed and robustness metrics.

---

### Rank 20 — Similarity: 0.8748
**Title**: Counterfactual Graph Learning for Link Prediction
**Year**: 2022 | **Decision**: Reject | **Submission**: YxQiIOLKgEf
**Citation percentile**: 0.55 | **Ratings**: [5, 8, 6, 5]

**True Abstract**:
Learning to predict missing links is important for many graph-based applications. Existing methods were designed to learn the association between two sets of variables: (1) the observed graph structure (e.g., clustering effect) and (2) the existence of link between a pair of nodes. However, the causal relationship between these variables was ignored. We visit the possibility of learning it by asking a counterfactual question: “would the link exist or not if the observed graph structure became different?” To answer this question, we leverage causal models considering the information of the node pair (i.e., learned graph representations) as context, global graph structural properties as treatment, and link existence as outcome. In this work, we propose a novel link prediction method that enhances graph learning by counterfactual inference. It creates counterfactual links from the observed ones, and learns representations from both the observed and counterfactual links. Experiments on benchmark datasets show that this novel graph learning method achieves state-of-the-art performance on link prediction.

**Generated Abstract**:
We introduce a novel framework for link prediction in graphs by leveraging counterfactual reasoning. Our method generates counterfactual instances to learn more robust representations that capture the underlying structural and semantic properties of the graph. By contrasting these counterfactual instances with the original graph data, we enhance the model's ability to predict missing or potential links while mitigating the effects of data bias and noise. Experimental results on various real-world datasets demonstrate significant improvements in link prediction accuracy compared to state-of-the-art methods.

---

## Bottom 10 Least Similar

### Rank 1 (from bottom) — Similarity: 0.2450
**Title**: Let's Verify Step by Step
**Year**: 2024 | **Decision**: Accept | **Submission**: v8L0pN6EOi

**True Abstract**:
In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.

**Generated Abstract**:
This paper proposes a novel framework for step-by-step verification of complex systems, focusing on breaking down the verification process into manageable, sequential components to enhance clarity and reliability. The approach is designed to systematically identify and resolve potential issues at each stage, thereby ensuring comprehensive and accurate validation of the system's functionality and integrity. Case studies from various domains demonstrate the effectiveness of the method in reducing verification time and increasing confidence in system reliability.

---

### Rank 2 (from bottom) — Similarity: 0.3264
**Title**: Unearthing Skill-level Insights for Understanding Trade-offs of Foundation Models
**Year**: 2025 | **Decision**: Accept | **Submission**: kNHVViEPWK

**True Abstract**:
With models getting stronger, evaluations have grown more complex, testing multiple skills in one benchmark and even in the same instance at once. However, skill-wise performance is obscured when inspecting aggregate accuracy, under-utilizing the rich signal modern benchmarks contain. We propose an automatic approach to recover the underlying skills relevant for any evaluation instance, by way of inspecting model-generated {\em rationales}. After validating the relevance of rationale-parsed skills and inferring skills for $46$k instances over $12$ benchmarks, we observe many skills to be common across benchmarks, resulting in the curation of hundreds of \emph{skill-slices} (i.e. sets of instances testing a common skill). Inspecting accuracy over these slices yields novel insights on model trade-offs: e.g., compared to GPT-4o and Claude 3.5 Sonnet, on average, Gemini 1.5 Pro is $18\%$ more accurate in \emph{computing molar mass}, but $19\\%$ less accurate in \emph{applying constitutional law}, despite the overall accuracies of the three models differing by a mere $0.4\\%$. Furthermore, we demonstrate the practical utility of our approach by showing that insights derived from skill slice analysis can generalize to held-out instances: when routing each instance to the model strongest on the relevant skills, we see a $3\\%$ accuracy improvement over our $12$ dataset corpus. Our skill-slices and framework open a new avenue in model evaluation, leveraging skill-specific analyses to unlock a more granular and actionable understanding of model capabilities.

**Generated Abstract**:
This paper explores the skill-level insights of foundation models across various domains to elucidate the trade-offs inherent in their performance. By analyzing a diverse set of benchmarks and tasks, we uncover how different foundation models vary in their strengths and weaknesses, providing a nuanced understanding of their capabilities and limitations. These insights are crucial for selecting the most appropriate model for specific applications and for guiding the future development of more versatile and efficient foundation models.

---

### Rank 3 (from bottom) — Similarity: 0.3310
**Title**: Learning to Link
**Year**: 2020 | **Decision**: Accept | **Submission**: S1eRbANtDB

**True Abstract**:
Clustering is an important part of many modern data analysis pipelines, including network analysis and data retrieval. There are many different clustering algorithms developed by various communities, and it is often not clear which algorithm will give the best performance on a specific clustering task. Similarly, we often have multiple ways to measure distances between data points, and the best clustering performance might require a non-trivial combination of those metrics. In this work, we study data-driven algorithm selection and metric learning for clustering problems, where the goal is to simultaneously learn the best algorithm and metric for a specific application. The family of clustering algorithms we consider is parameterized linkage based procedures that includes single and complete linkage. The family of distance functions we learn over are convex combinations of base distance functions. We design efficient learning algorithms which receive samples from an application-specific distribution over clustering instances and learn a near-optimal distance and clustering algorithm from these classes. We also carry out a comprehensive empirical evaluation of our techniques showing that they can lead to significantly improved clustering performance on real-world datasets.

**Generated Abstract**:
This paper presents a novel approach to address the challenge of entity linking in natural language processing, where the system learns to associate mentions in text with their corresponding entities in a knowledge base. We introduce a deep learning framework that leverages both local and global context to improve the accuracy of linking. Experiments on several benchmark datasets demonstrate that our method outperforms existing state-of-the-art techniques, achieving significant improvements in linking precision and coverage. Additionally, we explore the impact of various architectural choices and propose strategies to enhance the robustness of the model across different domains and languages.

---

### Rank 4 (from bottom) — Similarity: 0.3343
**Title**: Proactive Sequence Generator via Knowledge Acquisition
**Year**: 2020 | **Decision**: Reject | **Submission**: rJehf0VKwS

**True Abstract**:
Sequence-to-sequence models such as transformers, which are now being used in a wide variety of NLP tasks, typically need to have very high capacity in order to perform well. Unfortunately, in production, memory size and inference speed are all strictly constrained. To address this problem, Knowledge Distillation (KD), a technique to train small models to mimic larger pre-trained models, has drawn lots of attention. The KD approach basically attempts to maximize recall, i.e., ranking Top-k”tokens in teacher models as higher as possible, however, whereas precision is more important for sequence generation because of exposure bias. Motivated by this, we develop Knowledge Acquisition (KA) where student models receive log q(y_t|y_{<t},x) as rewards when producing the next token y_t given previous tokens y_{<t} and the source sentence x. We demonstrate the effectiveness of our approach on WMT’17 De-En and IWSLT’15 Th-En translation tasks, with experimental results showing that our approach gains +0.7-1.1 BLEU score compared to token-level knowledge distillation.

**Generated Abstract**:
This paper presents a novel approach to creating a proactive sequence generator that leverages knowledge acquisition techniques to enhance predictive capabilities and adapt to dynamic environments. The proposed method dynamically gathers and integrates domain-specific knowledge to generate sequences of actions or events that anticipate future needs or trends, thereby improving the efficiency and effectiveness of decision-making processes. Experimental results demonstrate that the proactive sequence generator outperforms traditional reactive systems in various scenarios, showcasing its potential applications in fields such as automated planning, recommendation systems, and intelligent agent design.

---

### Rank 5 (from bottom) — Similarity: 0.3505
**Title**: Provably Filtering Exogenous Distractors using Multistep Inverse Dynamics
**Year**: 2022 | **Decision**: Accept | **Submission**: RQLLzMCefQu

**True Abstract**:
Many real-world applications of reinforcement learning (RL) require the agent to deal with high-dimensional observations such as those generated from a megapixel camera. Prior work has addressed such problems with representation learning, through which the agent can provably extract endogenous, latent state information from raw observations and subsequently plan efficiently. However, such approaches can fail in the presence of temporally correlated noise in the observations, a phenomenon that is common in practice. We initiate the formal study of latent state discovery in the presence of such exogenous noise sources by proposing a new model, the Exogenous Block MDP (EX-BMDP), for rich observation RL. We start by establishing several negative results, by highlighting failure cases of prior representation learning based approaches. Then, we introduce the Predictive Path Elimination (PPE) algorithm, that learns a generalization of inverse dynamics and is provably sample and computationally efficient in EX-BMDPs when the endogenous state dynamics are near deterministic. The sample complexity of PPE depends polynomially on the size of the latent endogenous state space while not directly depending on the size of the observation space, nor the exogenous state space. We provide experiments on challenging exploration problems which show that our approach works empirically.

**Generated Abstract**:
This paper presents a novel approach for filtering exogenous distractors in robotic manipulation tasks using multistep inverse dynamics. By leveraging predictive models of the robot's dynamics, the method is able to anticipate and mitigate the effects of external disturbances, thereby improving task performance and robustness. We develop a theoretical framework that ensures the filter's effectiveness under certain conditions and validate our approach through extensive simulations and real-world experiments. The results demonstrate significant improvements in the robot's ability to maintain precise control and achieve its goals in the presence of distractors, contributing to a more reliable and adaptable robotic manipulation system.

---

### Rank 6 (from bottom) — Similarity: 0.3634
**Title**: Information-Theoretic Diffusion
**Year**: 2023 | **Decision**: Accept | **Submission**: UvmDCdSPDOW

**True Abstract**:
Denoising diffusion models have spurred significant gains in density modeling and image generation, precipitating an industrial revolution in text-guided AI art generation. We introduce a new mathematical foundation for diffusion models inspired by classic results in information theory that connect Information with Minimum Mean Square Error regression, the so-called I-MMSE relations. We generalize the I-MMSE relations to \emph{exactly} relate the data distribution to an optimal denoising regression problem, leading to an elegant refinement of existing diffusion bounds. This new insight leads to several improvements for probability distribution estimation, including a theoretical justification for diffusion model ensembling. Remarkably, our framework shows how continuous and discrete probabilities can be learned with the same regression objective, avoiding domain-specific generative models used in variational methods.

**Generated Abstract**:
This paper introduces a novel information-theoretic framework for analyzing and optimizing diffusion processes in complex networks. By leveraging concepts from information theory, such as entropy and mutual information, we develop metrics and algorithms to quantify and enhance the spread of information, innovations, or behaviors across networked systems. Our approach provides a unified perspective on diffusion phenomena, offering insights into the underlying mechanisms that govern their dynamics. We demonstrate the effectiveness of our framework through empirical analysis on various real-world networks, showing improvements in diffusion efficiency and coverage. The results have implications for a wide range of applications, including social media marketing, epidemiological modeling, and the spread of technological innovations.

---

### Rank 7 (from bottom) — Similarity: 0.3702
**Title**: Rapid Task-Solving in Novel Environments
**Year**: 2021 | **Decision**: Accept | **Submission**: F-mvpFpn_0q

**True Abstract**:
We propose the challenge of rapid task-solving in novel environments (RTS), wherein an agent must solve a series of tasks as rapidly as possible in an unfamiliar environment. An effective RTS agent must balance between exploring the unfamiliar environment and solving its current task, all while building a model of the new environment over which it can plan when faced with later tasks. While modern deep RL agents exhibit some of these abilities in isolation, none are suitable for the full RTS challenge. To enable progress toward RTS, we introduce two challenge domains: (1) a minimal RTS challenge called the Memory&Planning Game and (2) One-Shot StreetLearn Navigation, which introduces scale and complexity from real-world data. We demonstrate that state-of-the-art deep RL agents fail at RTS in both domains, and that this failure is due to an inability to plan over gathered knowledge. We develop Episodic Planning Networks (EPNs) and show that deep-RL agents with EPNs excel at RTS, outperforming the nearest baseline by factors of 2-3 and learning to navigate held-out StreetLearn maps within a single episode. We show that EPNs learn to execute a value iteration-like planning algorithm and that they generalize to situations beyond their training experience.

**Generated Abstract**:
This paper explores the mechanisms underlying rapid task-solving in novel environments, focusing on the interplay between cognitive flexibility, learning strategies, and neural adaptability. We present experimental evidence from both human and animal subjects demonstrating that individuals and animals can efficiently adapt to new tasks with minimal prior experience. Our findings suggest that a combination of prefrontal cortex-mediated cognitive processes and basal ganglia-driven habit formation plays a crucial role in this ability. Furthermore, we discuss the implications of these results for understanding the cognitive and neural bases of rapid learning and suggest potential applications in education, robotics, and artificial intelligence.

---

### Rank 8 (from bottom) — Similarity: 0.3868
**Title**: Efficiently Scanning and Resampling Spatio-Temporal Tasks with Irregular Observations
**Year**: 2025 | **Decision**: Reject | **Submission**: TySMCLoGVl

**True Abstract**:
Various works have aimed at combining the inference efficiency of recurrent models and training parallelism of MHA for sequence modeling. However, most of these works focus on tasks with fixed-dimension observation spaces, such as individual tokens in language modeling or pixels in image completion. Variably sized, irregular observation spaces are relatively under-represented, yet they occur frequently in multi-agent domains such as autonomous driving and human-robot interaction. To handle an observation space of varying size, we propose a novel algorithm that alternates between cross-attention between a 2D latent state and observation, and a discounted cumulative sum over the sequence dimension to efficiently accumulate historical information. We find this resampling cycle is critical for performance. To evaluate efficient sequence modeling in this domain, we introduce two multi-agent intention tasks: simulated agents chasing bouncing particles and micromanagement analysis in professional StarCraft II games. Our algorithm achieves comparable accuracy with a lower parameter count, faster training and inference compared to existing methods.

**Generated Abstract**:
This paper presents a novel algorithm for efficiently scanning and resampling spatio-temporal tasks, particularly those involving irregularly observed data. The method addresses the computational challenges associated with processing complex spatio-temporal datasets by leveraging advanced interpolation techniques and adaptive resampling strategies. It aims to provide a robust framework for handling tasks where data points are not uniformly distributed in space and time, ensuring both accuracy and computational efficiency. Experimental results demonstrate significant improvements in processing speed and resource utilization compared to existing methods, while maintaining or enhancing the quality of the resampled data. The approach is validated through various real-world applications, including environmental monitoring and traffic flow analysis, showcasing its versatility and practical utility in different domains.

---

### Rank 9 (from bottom) — Similarity: 0.3890
**Title**: Watch Out!! Your Confidence Might be a Reason for Vulnerability
**Year**: 2025 | **Decision**: Reject | **Submission**: 0IqriWHWYy

**True Abstract**:
The tremendous success of deep neural networks (DNNs) in solving `any' complex computer vision task leaves no stone unturned for their deployment in the physical world. However, the concerns arise when natural adversarial corruptions might perturb the physical world in unconstrained images. It is widely known that these corruptions are inherently present in the environment and can fool DNNs. While the literature aims to provide safety to DNNs against these natural corruptions they have developed two forms of defenses: (i) detection of corrupted images and (ii) mitigation of corruptions. So far, very little work has been done to understand the reason behind the vulnerabilities of DNNs against such corruption. We assert that network confidence is an essential component and ask whether the higher it is, the better the decision of a network is or not. Moreover, we ask the question of whether this confidence itself is a reason for their vulnerability against corruption. We extensively study the correlation between the confidence of a model and its robustness in handling corruption. Through extensive experimental evaluation using multiple datasets and models, we found a significant connection between the confidence and robustness of a network.

**Generated Abstract**:
This paper explores the paradoxical relationship between confidence and vulnerability in the context of cybersecurity. It argues that individuals and organizations often underestimate risks due to overconfidence in their security measures, leading to increased susceptibility to cyber threats. Through a series of empirical studies and case analyses, the research demonstrates how misplaced confidence can create blind spots and facilitate cyber attacks. The findings suggest that fostering a culture of humility and continuous learning in cybersecurity practices is essential for enhancing overall security resilience.

---

### Rank 10 (from bottom) — Similarity: 0.4190
**Title**: A Dual-Modal Framework Utilizing Visual Prompts for Enhanced Patch Analysis
**Year**: 2025 | **Decision**: Reject | **Submission**: OXIIFZqiiN

**True Abstract**:
Patch representation learning has emerged as a crucial innovation in software development, leveraging machine learning techniques to advance software generation workflows. This approach has led to significant enhancements across various applications involving code alterations. However, existing methods often exhibit a tendency towards specialization, excelling predominantly in either predictive tasks such as security patch classification or in generative tasks like the automated creation of patch descriptions. This paper presents a groundbreaking approach to patch representation learning through the Image-Guided Code Patch Framework (IGCP), a novel architecture that bridges the gap between code analysis and image processing domains. We introduce a rigorous mathematical foundation for IGCP, leveraging measure theory, functional analysis, and information geometry to formalize the domain adaptation process in patch representation learning. The optimization dynamics of IGCP are rigorously analyzed through the lens of Stochastic Gradient Langevin Dynamics, providing convergence guarantees in both convex and non-convex loss landscapes. Empirical evaluations demonstrate that IGCP not only achieves state-of-the-art performance in patch description generation but also exhibits remarkable domain generalization capabilities.

**Generated Abstract**:
This paper introduces a novel dual-modal framework that integrates visual prompts to improve the accuracy and efficiency of patch analysis in various applications, such as image segmentation and anomaly detection. By leveraging both visual and textual information, our approach enhances the interpretability and robustness of the analysis process. We propose a method that uses visual prompts to guide the attention of the model towards relevant regions, thereby reducing the computational complexity and improving performance. Experimental results demonstrate significant improvements in task-specific metrics compared to existing single-modal approaches, particularly in challenging scenarios where visual context is crucial.

---

## Median Similarity Examples (5 around median)

### Median 1 — Similarity: 0.7295
**Title**: TorchTitan: One-stop PyTorch native solution for production ready LLM pretraining
**Year**: 2025 | **Decision**: Accept | **Submission**: SFN6Wm7YBI

**True Abstract**:
The development of large language models (LLMs) has been instrumental in advancing state-of-the-art natural language processing applications. Training LLMs with billions of parameters and trillions of tokens requires sophisticated distributed systems that enable composing and comparing several state-of-the-art techniques in order to efficiently scale across thousands of accelerators. However, existing solutions are complex, scattered across multiple libraries/repositories, lack interoperability, and are cumbersome to maintain. Thus, curating and empirically comparing training recipes requires non-trivial engineering effort. This paper introduces **TORCHTITAN**$^1$, a PyTorch-native distributed training system that unifies and advances state-of-the-art techniques, streamlining integration and reducing engineering overhead. TORCHTITAN enables seamless application of 4D parallelism in a modular and composable manner, while featuring elastic scaling to adapt to changing computational requirements. The system provides comprehensive logging, efficient checkpointing, and debugging tools, ensuring production-ready training. Moreover, TORCHTITAN incorporates innovative hardware-software co-designed solutions, leveraging cutting-edge features like Float8 training and SymmetricMemory to maximize hardware utilization. As a flexible experimental test bed, TORCHTITAN facilitates the curation and comparison of custom recipes for diverse training contexts. By leveraging TORCHTITAN, we developed optimized training recipes for the Llama 3.1 family and provide actionable guidance on selecting and combining distributed training techniques to maximize training efficiency, based on our hands-on experiences. We thoroughly assess TORCHTITAN on the Llama 3.1 family of LLMs, spanning 8 billion to 405 billion parameters, and showcase its exceptional performance, modular composability, and elastic scalability. By stacking training optimizations, we demonstrate accelerations ranging from 65.08% on Llama 3.1 8B at 128 GPU scale (1D), 12.59% on Llama 3.1 70B at 256 GPU scale (2D), to 30% on Llama 3.1 405B at 512 GPU scale (3D) on NVIDIA H100 GPUs over optimized baselines. We also demonstrate the effectiveness of 4D parallelism in enabling long context training.

**Generated Abstract**:
TorchTitan is a comprehensive, PyTorch-native framework designed to streamline the process of pretraining large language models (LLMs) for production readiness. This solution addresses the critical challenges in deploying LLMs by offering a unified interface for data preprocessing, model training, and inference optimization. TorchTitan supports a wide range of state-of-the-art pretraining techniques and provides scalable infrastructure to handle large datasets and complex model architectures. Additionally, it includes features for efficient resource management, automated hyperparameter tuning, and integration with popular cloud platforms, ensuring that LLMs can be seamlessly transitioned from research to production environments.

---

### Median 2 — Similarity: 0.7294
**Title**: Towards Equivariant Graph Contrastive Learning via Cross-Graph Augmentation
**Year**: 2023 | **Decision**: Reject | **Submission**: 9L1Ts8t66YK

**True Abstract**:
Leading graph contrastive learning (GCL) frameworks conform to the invariance mechanism by encouraging insensitivity to different augmented views of the same graph. Despite the promising performance, invariance worsens representation when augmentations cause aggressive semantics shifts. For example, dropping the super-node can dramatically change a social network's topology. In this case, encouraging invariance to the original graph can bring together dissimilar patterns and hurt the task of instance discrimination. To resolve the problem, we get inspiration from equivariant self-supervised learning and propose Equivariant Graph Contrastive Learning (E-GCL) to encourage the sensitivity to global semantic shifts. Viewing each graph as a transformation to others, we ground the equivariance principle as a cross-graph augmentation -- graph interpolation -- to simulate global semantic shifts. Without using annotation, we supervise the representation of cross-graph augmented views by linearly combining the representations of their original samples. This simple but effective equivariance principle empowers E-GCL with the ability of cross-graph discrimination. It shows significant improvements over the state-of-the-art GCL models in unsupervised learning and transfer learning. Further experiments demonstrate E-GCL's generalization to various graph pre-training frameworks.

**Generated Abstract**:
Graph contrastive learning (GCL) has emerged as a powerful technique for improving the representation learning of graph neural networks (GNNs) by leveraging positive and negative graph pairs. However, the existing GCL methods often struggle to capture the intrinsic equivariance properties of GNNs, which are crucial for handling permutation invariant data. This paper proposes a novel approach, named Cross-Graph Augmentation (CGA), to explicitly enforce equivariance in GCL. By augmenting graphs from different distributions, CGA generates a diverse set of positive and negative pairs that not only enhance the discriminative power of the learned representations but also ensure that the representations are equivariant to graph permutations. We evaluate our method on various graph classification tasks and demonstrate significant improvements in both representation quality and model performance, especially on datasets with complex structural variations. Our findings suggest that CGA is a promising direction for developing more robust and interpretable graph representations in the context of GCL.

---

### Median 3 — Similarity: 0.7293
**Title**: Mockingbird: Platform for Adapting LLMs to General Machine Learning Tasks
**Year**: 2025 | **Decision**: Reject | **Submission**: cLTM1gc6Qm

**True Abstract**:
Large language models (LLMs) are now being used with increasing frequency as chat bots, tasked with the summarizing information or generating text and code in accordance with user instructions. The rapid increase in reasoning capabilities and inference speed of LLMs has revealed their remarkable potential for applications extending beyond the domain of chat bots. However, there is a paucity of research exploring the integration of LLMs into a broader range of intelligent software systems. In this research, we propose a paradigm for leveraging LLMs as mock functions to adapt LLMs to general machine learning tasks. Furthermore, we present an implementation of this paradigm, entitled the Mockingbird platform. In this paradigm, users define mock functions which are defined solely by method signature and documentation. Unlike LLM-based code completion tools, this platform does not generate code at compile time; instead, it instructs the LLM to role-play these mock functions at runtime. Based on the feedback from users or error from software systems, this platform will instruct the LLM to conduct chains of thoughts to reflect on its previous output, thereby enabling it to perform reinforcement learning. This paradigm fully exploits the intrinsic knowledge and in-context learning ability of LLMs. In comparison to conventional machine learning methods, following distinctive advantages are offered: (a) Its intrinsic knowledge enables it to perform well in a wide range of zero-shot scenarios. (b) Its flexibility allows it to adapt to random increases or decreases of data fields. (c) It can utilize tools and extract information from sources that are inaccessible to conventional machine learning methods, such as the Internet. Finally, we evaluated its performance and demonstrated the previously mentioned benefits using several datasets from Kaggle. Our results indicate that this paradigm is highly competitive.

**Generated Abstract**:
Mockingbird is a novel platform designed to facilitate the adaptation of Large Language Models (LLMs) to a wide array of general machine learning tasks beyond natural language processing. This platform introduces a modular architecture that allows for seamless integration of LLMs with various machine learning techniques, enabling users to leverage the advanced text generation capabilities of LLMs for tasks such as image classification, time series prediction, and reinforcement learning. By providing a unified interface and a set of customizable components, Mockingbird aims to democratize access to LLMs and accelerate the development of cross-modal machine learning applications. Experimental results demonstrate that Mockingbird can effectively enhance the performance of machine learning models across different domains, showcasing the potential of LLMs in extending their utility beyond traditional language-centric tasks.

---

### Median 4 — Similarity: 0.7293
**Title**: Do Pre-trained Transformers Really Learn In-context by Gradient Descent?
**Year**: 2024 | **Decision**: Reject | **Submission**: 992eLydH8G

**True Abstract**:
Is In-Context Learning (ICL) implicitly equivalent to Gradient Descent (GD)? Several recent works draw analogies between the dynamics of GD and the emergent behavior of ICL in large language models. However, these works make assumptions far from the realistic natural language setting in which language models are trained. Such discrepancies between theory and practice, therefore necessitate further investigation to validate their applicability in reality. We start by highlighting the weaknesses in prior works that construct Transformer weights to simulate gradient descent. Their experiments with training Transformers on ICL objective, inconsistencies in the order-sensitivity of ICL and GD, sparsity of the constructed weights, and sensitivity to parameter changes are some examples of a mismatch from the real-world setting. Furthermore, we probe and compare the ICL vs. GD hypothesis in a natural setting. We conduct comprehensive empirical analyses on language models pre-trained on natural data (LLaMa-7B). Our comparisons on various performance metrics highlight the inconsistent behavior of ICL and GD as a function of various factors such as datasets, models, and number of demonstrations. We observe that ICL and GD adapt the output distribution of language models differently. These results indicate that the equivalence between ICL and GD is an open hypothesis, requires nuanced considerations and calls for further studies.

**Generated Abstract**:
This paper investigates the mechanism by which pre-trained transformers utilize in-context examples to solve new tasks, questioning whether this capability is genuinely achieved through gradient descent. Through a series of empirical analyses and controlled experiments, we demonstrate that the effectiveness of pre-trained transformers in leveraging in-context information may not solely rely on gradient descent during fine-tuning. Instead, we find evidence suggesting that these models might be employing in-context examples in ways that are more aligned with human-like learning, potentially through mechanisms such as meta-learning or by utilizing stored information from their pre-training phase. Our findings challenge the prevailing assumption that gradient descent is the primary driver of in-context learning in these models and suggest the need for a more nuanced understanding of their learning dynamics.

---

### Median 5 — Similarity: 0.7293
**Title**: Cascade Reward Sampling for Efficient Decoding-Time Alignment
**Year**: 2025 | **Decision**: Reject | **Submission**: UAA2nWUtVl

**True Abstract**:
Aligning large language models (LLMs) with human preferences is critical for their deployment. Recently, decoding-time alignment has emerged as an effective plug-and-play technique that requires no fine-tuning of model parameters. However, generating text that achieves both high reward and high likelihood remains a significant challenge. Existing methods often fail to generate high-reward text or incur substantial computational costs. In this paper, we propose Cascade Reward Sampling (CARDS) to address both issues, guaranteeing the generation of high-reward and high-likelihood text with significantly low costs. Based on our analysis of reward models (RMs) on incomplete text and our observation that high-reward prefixes induce high-reward complete text, we use rejection sampling to iteratively generate small semantic segments to form such prefixes. The segment length is dynamically determined by the predictive uncertainty of LLMs. This strategy guarantees desirable prefixes for subsequent generations and significantly reduces wasteful token re-generations and the number of reward model scoring. Our experiments demonstrate substantial gains in both generation efficiency and alignment ratings compared to the baselines, achieving five times faster text generation and 99% win-ties in GPT-4/Claude-3 helpfulness evaluation.

**Generated Abstract**:
We introduce Cascade Reward Sampling (CRS), a novel method designed to enhance the efficiency of decoding-time alignment in sequence-to-sequence models. CRS addresses the challenge of balancing exploration and exploitation during training by adaptively sampling rewards from multiple cascaded subtasks, leading to improved alignment accuracy and reduced computational overhead. Through extensive experiments on various alignment tasks, we demonstrate that CRS outperforms existing methods in terms of both speed and alignment quality, making it a promising approach for applications requiring real-time or near-real-time decoding.

---
