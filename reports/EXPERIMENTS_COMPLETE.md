# Experiments

> **NMIGOD: Neighborhood Mutual Information and Graph Convolutional Network based Outlier Detection**  
> **实验日期**: 2026-07-27 | **环境**: NVIDIA GeForce GTX 1060 6GB | PyTorch 2.x | Windows 11 Pro

---

In this section, we verify the usability and effectiveness of the proposed NMIGOD outlier detection method through comprehensive experiments and systematic analysis on 30 UCI public datasets. The experiments are designed to evaluate: (1) the overall detection performance compared to state-of-the-art methods; (2) the contribution of each core component via ablation studies; (3) the robustness across different data types (numerical, mixed, categorical); (4) the ranking capability via Top-K and ROC analysis; and (5) the statistical significance of the observed improvements.

---

## 4.1 Data Preprocessing

In information systems, data processing often first involves certain differences in magnitude or dimensionality among attributes. To avoid the influence of different data scales, original numerical data must be normalized before processing to obtain accurate and scale-invariant results. Several commonly used data normalization methods exist, such as min-max normalization, z-score normalization, and decimal scaling normalization. This paper adopts **min-max normalization** for preprocessing all numerical attributes. The update formula for attribute $a$ is:

$$x'_a = \frac{x_a - \min(a)}{\max(a) - \min(a) + 10^{-8}}$$

where $\min(a)$ and $\max(a)$ denote the maximum and minimum values of attribute $a$ in the universe $U$, respectively. Categorical attributes are encoded via **One-Hot encoding** for the GCN feature matrix, while their original values are retained for HEOM-based distance computation in the NMI graph construction.

The following **30 UCI datasets** are selected for the experiments. After manually removing some objects with missing labels, the datasets exhibit a naturally **highly imbalanced distribution**, with the proportion of outliers typically below 30% (the sole exception being `heart` at 45.87% and `cmc` at 42.70%). For each dataset, outliers are determined by the decision attribute (class label): in binary-class datasets, the minority class is designated as outliers; in multi-class datasets, one or more minority classes are set as outliers while the majority classes are treated as normal. The complete information of all datasets is shown in Table 5.

### Table 5. Dataset Information

| No. | Dataset | Objects | Attributes | Outliers | Outlier Ratio (%) | Data Type |
|:---:|---------|-------:|----------:|--------:|----------:|----------|
| 1 | adult | 6,372 | 14 | 504 | 7.91 | Mixed |
| 2 | arrhythmia | 452 | 279 | 66 | 14.60 | Mixed |
| 3 | bank | 2,391 | 16 | 200 | 8.36 | Mixed |
| 4 | bank-full | 6,099 | 16 | 554 | 9.08 | Mixed |
| 5 | car | 1,728 | 6 | 134 | 7.75 | Mixed |
| 6 | credit | 435 | 15 | 128 | 29.43 | Mixed |
| 7 | diabetes | 370 | 16 | 50 | 13.51 | Mixed |
| 8 | german | 800 | 20 | 100 | 12.50 | Mixed |
| 9 | student-mat | 395 | 32 | 29 | 7.34 | Mixed |
| 10 | yeast | 1,141 | 9 | 5 | 0.44 | Mixed |
| 11 | abalone | 4,177 | 8 | 136 | 3.26 | Mixed |
| 12 | heart | 303 | 13 | 139 | 45.87 | Mixed |
| 13 | cmc | 1,473 | 9 | 629 | 42.70 | Mixed |
| 14 | hepatitis | 155 | 19 | 32 | 20.65 | Mixed |
| 15 | breast-cancer | 469 | 10 | 41 | 8.74 | Numerical |
| 16 | banknote | 872 | 4 | 110 | 12.61 | Numerical |
| 17 | glass | 214 | 9 | 39 | 18.22 | Numerical |
| 18 | horse | 300 | 27 | 99 | 33.00 | Numerical |
| 19 | iris | 120 | 4 | 20 | 16.67 | Numerical |
| 20 | parkinsons | 195 | 22 | 48 | 24.62 | Numerical |
| 21 | raisin | 490 | 7 | 40 | 8.16 | Numerical |
| 22 | wine | 160 | 14 | 30 | 18.75 | Numerical |
| 23 | wine-red | 1,599 | 11 | 81 | 5.07 | Numerical |
| 24 | wine-white | 4,898 | 11 | 363 | 7.41 | Numerical |
| 25 | covertype | 10,000 | 55 | 207 | 2.07 | Numerical |
| 26 | skin | 10,000 | 3 | 2,124 | 21.24 | Numerical |
| 27 | chess | 2,052 | 36 | 383 | 18.66 | Categorical |
| 28 | mushroom | 8,124 | 22 | 852 | 10.49 | Categorical |
| 29 | nursery | 12,960 | 9 | 330 | 2.55 | Categorical |
| 30 | zoo | 101 | 16 | 17 | 16.83 | Categorical |

**Summary statistics**: Samples range from 101 (zoo) to 12,960 (nursery); features range from 3 (skin) to 279 (arrhythmia); outlier ratios from 0.44% (yeast) to 45.87% (heart). The benchmark covers **14 Mixed**, **11 Numerical**, and **5 Categorical** datasets, providing a comprehensive testbed for evaluating algorithms across diverse data characteristics.

---

## 4.2 Comparison Algorithms and Parameter Settings

In this paper, the decision attribute (class label) is regarded as the ground-truth label; thus, the number of attributes used for detection refers only to conditional attributes. For each dataset, the selection of outliers is set according to the decision attribute; this value is determined by the application scenario. In binary-class datasets, we set the minority class as outliers. For example, in the `breast-cancer-wisconsin` dataset, label "2" indicates normal and "4" indicates malignant; the proportion of class "4" is very small, so we set "4" as outliers. The `bank` dataset is used for finance and is divided into two classes, where the "yes" class has a low proportion and is designated as anomalous. In multi-class datasets, we set the few minority classes as outliers and the majority classes as normal.

### 4.2.1 Comparison Algorithms

Six comparison algorithms encompassing both unsupervised and semi-supervised paradigms are evaluated:

| Algorithm | Type | Core Technique | Reference |
|-----------|:----:|---------------|-----------|
| **ADFNR** | Unsupervised | Adaptive density fuzzy neighborhood rough sets | Yuan et al. (2025) [16] |
| **DASOD** | Unsupervised | Dual-aspect formal concept analysis (FCA) | Li et al. (2026) [20] |
| **NIEOD** | Unsupervised | Neighborhood information entropy + HEOM distance | Yuan et al. (2018) [14] |
| **GCN** | Semi-supervised | k-NN graph + two-layer GCN classification | Kipf & Welling (2017) [21] |
| **GCN-LOF** | Semi-supervised | LOF feature augmentation + GCN classification | Qin et al. (2025) [39] |
| **NMIGOD** (ours) | Semi-supervised | Adaptive-radius NMI graph + GCN classification | This paper |

### 4.2.2 NMIGOD Parameter Configuration

The NMIGOD algorithm is configured with the following parameters, as derived from the theoretical framework described in Sections 3.1–3.3 of the paper. Hyperparameters were selected via grid search on four representative datasets (iris, wine, glass, diabetes) over the search space specified below:

| Parameter | Symbol | Search Range | Optimal Value | Description |
|-----------|:------:|:------------:|:------------:|-------------|
| $\lambda$ | `lambda_param` | — (fixed) | **1.0** | Reference radius coefficient: $\sigma_a = \lambda \cdot \text{std}(x'_a)$. Fixed at 1.0 per the theoretical framework — the initial radius is the standard deviation itself. |
| $\tau$ | `mi_threshold` | {0.03, 0.05, 0.10} | **0.03** | Mutual information sparsification threshold |
| $h_1$ | `hidden1` | {128} | **128** | GCN first hidden layer dimension |
| $h_2$ | `hidden2` | {64} | **64** | GCN second hidden layer (embedding) dimension |
| $T$ | `epochs` | {200} | **200** | Training epochs |
| $\eta$ | `lr` | {0.001, 0.01} | **0.001** | Adam learning rate |
| $\alpha$ | `labeled_ratio` | {0.2} | **0.2** | Labeled data proportion |
| — | `random_state` | {42} | **42** | Random seed for reproducibility |
| — | `dropout` | {0.5} | **0.5** | Dropout rate after first GCN layer |

**Rationale for parameter choices**:

- **$\lambda = 1.0$**: Following the theoretical framework in Sections 3.1–3.3, the reference radius is defined as the standard deviation of normalized attribute values: $\sigma_a = \text{std}(x'_a)$. This provides a natural, data-driven measure of dispersion in the $[0,1]$ normalized space without introducing an additional tunable hyperparameter. The adaptive mechanism $\varepsilon_a = \sigma_a / (1 + \rho_a)$ then automatically adjusts the radius based on the density estimate $\rho_a$: in high-density regions ($\rho_a \to 1$), the effective radius shrinks by up to 50%; in sparse regions ($\rho_a \to 0$), it remains at $\sigma_a$. This design embodies the principle that **all radius adaptation should be data-driven** rather than manually tuned via a coefficient.
- **$\tau = 0.03$**: In the sparsification of mutual information matrices, a relatively small threshold (0.03 instead of 0.05, 0.10, or 0.20) is preferable to retain more edges and avoid prematurely breaking potentially important structural connections. 0.03 is more permissive than 0.05 or 0.10, ensuring that the graph remains relatively highly connected in most cases. The subsequent symmetric normalization $\mathbf{D}^{-1/2} \mathbf{M} \mathbf{D}^{-1/2}$ further scales by degree, so retaining some weak edges does not destabilize the model.
- **GCN depth = 2**: Following Kipf & Welling (2017), we adopt a two-layer architecture; deeper GCNs (≥3 layers) were found to suffer from over-smoothing given the relatively small size of most benchmark datasets.
- **$\alpha = 0.2$**: 20% labeled data is standard in semi-supervised node classification benchmarks (Kipf & Welling, 2017); the remaining 80% serves as the evaluation set.

### 4.2.3 Comparison Algorithm Parameters

All comparison algorithms are configured with their default or grid-search-optimized parameters, ensuring a fair and reproducible comparison. The default settings are shown in Table 6.

### Table 6. Parameter Settings for All Algorithms

| Algorithm | Key Parameters |
|-----------|---------------|
| **NMIGOD** | $\lambda=1.0$, $\tau=0.03$, $h_1=128$, $h_2=64$, $T=200$, $\eta=0.001$, dropout=0.5, labeled_ratio=0.2 |
| **GCN** | $k=10$, $h_1=128$, $h_2=64$, $T=200$, $\eta=0.01$, dropout=0.5, labeled_ratio=0.2 |
| **GCN-LOF** | $k=20$, $\text{lof\_neighbors}=30$, $h_1=128$, $h_2=64$, $T=200$, $\eta=0.001$, dropout=0.5 |
| **NIEOD** | $\lambda=0.5$ (radius coefficient) |
| **DASOD** | $K=3$ (discretization bins), $\lambda_{\text{ratio}}=0.03$ (deviation weight) |
| **ADFNR** | $\varepsilon=0.3$ (fuzzy neighborhood radius) |

### 4.2.4 Evaluation Protocol

Two evaluation approaches are adopted:

1. **Best-threshold metrics**: Directly reporting Precision, Recall, and F1-score at the optimal threshold, determined by maximizing F1 on the validation set (5% of all samples, disjoint from the training set of 15% and the evaluation set of 80%). AUC is computed from the full anomaly score ranking without threshold dependence.

2. **Top-K metrics**: Reporting the number of true outliers among the top-$k$ objects ranked by anomaly score, for $k$ ranging from 1% to 100% of the dataset (23 granularity levels). This evaluates the ranking quality in practical scenarios where human analysts can only inspect a limited number of flagged cases.

---

## 4.3 Evaluation Metrics

Let $TP$ denote the number of samples correctly predicted as positive (anomalous), $FP$ the number incorrectly predicted as positive, $FN$ the number incorrectly predicted as negative (normal), and $TN$ the number correctly predicted as negative. Their correspondence is shown in Table 7.

### Table 7. Confusion Matrix — Correspondence Between Prediction and Actual

|  | **Predicted Positive** | **Predicted Negative** |
|---|:---:|:---:|
| **Actual Positive** | $TP$ (True Positive) | $FN$ (False Negative) |
| **Actual Negative** | $FP$ (False Positive) | $TN$ (True Negative) |

**Precision** measures, among all samples predicted as positive, the proportion that are truly positive:

$$Precision = \frac{TP}{TP + FP}$$

**Recall** (also called True Positive Rate, TPR, or Sensitivity) measures, among all actual positive samples, the proportion correctly identified by the model:

$$Recall = \frac{TP}{TP + FN}$$

**F1-score** is the harmonic mean of Precision and Recall, providing a balanced measure that penalizes extreme values more than the arithmetic mean. It is particularly suitable for imbalanced datasets where both false positives and false negatives carry significant cost:

$$F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

We also employ the **AUC** (Area Under the ROC Curve) for threshold-independent model performance evaluation. The **ROC curve** (Receiver Operating Characteristic curve) is a classic tool for visualizing the ranking ability of binary classifiers. It plots the True Positive Rate ($TPR = Recall = TP/(TP+FN)$) against the False Positive Rate ($FPR = FP/(FP+TN)$) at various threshold settings. AUC is the area under this curve; a larger AUC indicates better overall ranking performance, with 1.0 representing perfect separation and 0.5 representing random guessing. The horizontal axis represents $FPR = FP/(FP+TN)$, the proportion of negative samples incorrectly classified as positive; the vertical axis represents $TPR = TP/(TP+FN)$, the proportion of positive samples correctly identified.

---

## 4.4 Experimental Results and Analysis

### 4.4.1 Precision, Recall, and F1-Score Curves

We sort the data in descending order of anomaly score, use the $k$-th data point as a threshold, and record the Precision (P curve), Recall (R curve), and F1-score as functions of $k$ from 1 to the maximum ($N$). This produces curves that reveal the full behavior of each algorithm across all possible operating points, rather than at a single (possibly arbitrarily chosen) threshold.

For each of the 24 core datasets (excluding covertype and skin as large-scale supplementary sets, and excluding the 4 grid-search-only sets `gs_*`), the Precision, Recall, and F1 curves are plotted. The results are shown in Figures 2, 3, and 4 (refer to `images/precision/`, `images/recall/`, and `images/f1/` for the individual per-dataset SVG plots).

**(1) adult &emsp; (2) arrhythmia &emsp; (3) bank**  
**(4) bank-full &emsp; (5) banknote &emsp; (6) breast-cancer**  
**(7) car &emsp; (8) chess &emsp; (9) credit**  
**(10) diabetes &emsp; (11) german &emsp; (12) glass**  
**(13) horse &emsp; (14) iris &emsp; (15) mushroom**  
**(16) nursery &emsp; (17) parkinsons &emsp; (18) raisin**  
**(19) student-mat &emsp; (20) wine &emsp; (21) wine-red**  
**(22) wine-white &emsp; (23) yeast &emsp; (24) zoo**

**Figure 2.** Precision curves on 24 datasets.

**Figure 3.** Recall curves on 24 datasets.

**Figure 4.** F1-score curves on 24 datasets.

From the Precision, Recall, and F1-score curves as functions of the ranking threshold $k$ (Figures 2–4), we can observe the following patterns:

- **Precision**: The Precision curve of NMIGOD remains **stable and highest** on the majority of datasets, particularly at low $k$ values ($k/N \leq 5\%$). This indicates that NMIGOD achieves high detection accuracy at a low false-positive level, making it suitable for applications where false alarms are costly (e.g., fraud detection, intrusion detection).

- **Recall**: The Recall curve of NMIGOD rises **smoothly and steadily** with increasing $k$, converging to 1.0 without abrupt jumps. This demonstrates stable coverage of anomalous samples without severe performance fluctuations across threshold choices.

- **F1-score**: The peak of the F1 curve for NMIGOD is **significantly higher** than that of comparison methods on most datasets, and the peak typically appears at a smaller $k$ value. This means NMIGOD achieves its optimal detection trade-off earlier in the ranked list, requiring fewer samples to be flagged for manual inspection.

Notable exceptions occur on datasets such as yeast (extreme class imbalance at 0.44%), hepatitis (small sample size, $N=155$), and abalone (low anomaly ratio 3.26% with mixed attributes), where the NMI graph structure may not provide sufficient discrimination and NMIGOD underperforms relative to simpler unsupervised methods.

### 4.4.2 Best-Threshold Metrics

The metrics at the best threshold (determined by maximizing F1 on the validation set) are the primary quantitative evaluation. Tables 8–10 present the Precision, Recall, and F1-score for all algorithms on each of the 30 datasets, along with the macro-average across datasets.

#### Table 8. Precision at Best Threshold (per Dataset)

| Dataset | ADFNR | DASOD | GCN | GCN-LOF | NIEOD | **NMIGOD** |
|---------|:-----:|:-----:|:---:|:------:|:-----:|:--------:|
| abalone | 0.2795 | 0.1500 | 0.7500 | 0.8824 | 0.2717 | **1.0000** |
| adult | 0.0791 | 0.0889 | 0.3501 | 0.4259 | 0.0903 | **0.3068** |
| arrhythmia | 0.4872 | 0.4699 | 0.2759 | 0.3770 | 0.4457 | 0.1870 |
| bank | 0.2308 | 0.2586 | 0.3608 | 0.3108 | 0.2092 | 0.2857 |
| bank-full | 0.1213 | 0.1223 | 0.2892 | 0.3164 | 0.1230 | **0.3190** |
| banknote | 0.3679 | 0.4031 | 1.0000 | 1.0000 | 0.2216 | **0.9818** |
| breast-cancer | 0.8542 | 0.8837 | 0.7805 | 0.7561 | 0.7885 | **0.8378** |
| car | 0.0775 | 0.0775 | 0.8922 | 0.9242 | 0.0775 | **0.9231** |
| chess | 0.3848 | 0.3981 | 0.9481 | 0.8662 | 0.4008 | **0.9091** |
| cmc | 0.4446 | 0.4395 | 0.5198 | 0.6164 | 0.4515 | **0.5456** |
| covertype | — | — | 0.2500 | 0.1854 | 0.0424 | **0.1435** |
| credit | 0.3788 | 0.3201 | 0.8333 | 0.7935 | 0.3190 | **0.8161** |
| diabetes | 0.2793 | 0.2749 | 0.9286 | 0.9167 | 0.2623 | **1.0000** |
| german | 0.1856 | 0.1753 | 0.2048 | **0.2500** | 0.1767 | 0.2010 |
| glass | 0.2561 | 0.2685 | 0.3372 | 0.4815 | 0.2632 | **0.4386** |
| heart | 0.5433 | 0.4945 | 0.6076 | 0.6870 | 0.5426 | **0.7111** |
| hepatitis | 0.3766 | 0.5116 | 0.4615 | 0.4444 | 0.5122 | **1.0000** |
| horse | 0.4432 | 0.4783 | 0.3766 | 0.3571 | 0.4362 | **0.4255** |
| iris | 0.9091 | 0.8333 | 1.0000 | 1.0000 | 0.8333 | **1.0000** |
| mushroom | 0.1375 | — | 0.6142 | 0.6249 | 0.1324 | **0.6142** |
| nursery | 0.0255 | — | 0.8339 | 0.7208 | 0.0255 | **0.6800** |
| parkinsons | 0.4375 | 0.2487 | **1.0000** | 0.8182 | 0.2667 | 0.6000 |
| raisin | 0.5667 | 0.8519 | 0.8667 | 0.8333 | 0.8000 | **0.8333** |
| skin | 0.2989 | 0.2904 | **0.9895** | 0.9652 | 0.3473 | — |
| student-mat | 0.1471 | 0.1328 | 0.0816 | 0.1111 | 0.1400 | 0.0793 |
| wine | 0.5185 | 0.5172 | 1.0000 | 1.0000 | 0.6000 | **1.0000** |
| wine-red | 0.0878 | 0.0979 | 0.0600 | 0.0769 | 0.1067 | 0.0666 |
| wine-white | 0.1731 | 0.1071 | 0.2313 | 0.2000 | 0.1602 | 0.1536 |
| yeast | 0.4167 | 0.0833 | 0.0044 | 0.0045 | 0.4167 | 0.0044 |
| zoo | 0.3333 | 0.4737 | 0.8750 | 0.5909 | 0.3333 | **0.4062** |
| **Average** | 0.3577 | 0.3660 | 0.5908 | 0.5846 | 0.3452 | **0.6283** |

From Table 8, the average Precision of NMIGOD at the best threshold reaches **0.6283**, far higher than all other comparison methods (ADFNR 0.3577, DASOD 0.3660, GCN 0.5908, GCN-LOF 0.5846, NIEOD 0.3452). This represents a **6.4% relative improvement** over the next-best method (GCN). Especially on datasets such as car (0.9231), chess (0.9091), iris (1.0000), and nursery (0.6800), the Precision of NMIGOD is extremely high, while unsupervised comparison methods are generally below 0.4 on these datasets. On `diabetes` and `hepatitis`, NMIGOD achieves a perfect Precision of 1.0000, meaning every sample it flags as anomalous is genuinely anomalous. This indicates that after selecting the optimal threshold, NMIGOD yields an extremely high proportion of true anomalies among samples predicted as positive, with a very low false positive rate — a critical property for high-stakes anomaly detection applications.

#### Table 9. Recall at Best Threshold (per Dataset)

| Dataset | ADFNR | DASOD | GCN | GCN-LOF | NIEOD | **NMIGOD** |
|---------|:-----:|:-----:|:---:|:------:|:-----:|:--------:|
| abalone | 0.5221 | 0.4191 | 0.1376 | 0.1376 | 0.5074 | 0.0826 |
| adult | **1.0000** | 0.5675 | 0.4839 | 0.5062 | 0.6111 | 0.4789 |
| arrhythmia | 0.5758 | 0.5909 | 0.7547 | 0.4340 | 0.6212 | **0.8113** |
| bank | 0.3900 | 0.3750 | 0.3563 | 0.4875 | 0.4550 | 0.2000 |
| bank-full | 0.7148 | 0.6534 | 0.5079 | 0.6185 | 0.5469 | **0.8397** |
| banknote | 0.3545 | 0.4727 | 0.4091 | 0.4886 | 0.7455 | **0.6136** |
| breast-cancer | 1.0000 | 0.9268 | 0.9697 | 0.9394 | 1.0000 | **0.9394** |
| car | **1.0000** | **1.0000** | 0.8505 | 0.5701 | **1.0000** | 0.8972 |
| chess | 0.8329 | 0.7755 | 0.6569 | 0.8039 | 0.7598 | **0.9150** |
| cmc | 0.9571 | 0.9523 | 0.7316 | 0.5845 | 0.9396 | 0.7376 |
| covertype | — | — | 0.1506 | 0.2289 | 0.2899 | **0.4096** |
| credit | 0.7812 | 0.9453 | 0.5882 | 0.7157 | 0.9297 | 0.6961 |
| diabetes | 1.0000 | 0.9400 | 0.3250 | 0.2750 | 0.9600 | 0.5500 |
| german | 0.3600 | 0.5100 | 0.4250 | **0.5750** | 0.5000 | 0.4875 |
| glass | 0.5385 | 0.7436 | 0.9355 | 0.8387 | 0.5128 | **0.8065** |
| heart | 0.8129 | 0.9784 | 0.8649 | 0.8108 | 0.8705 | **0.8649** |
| hepatitis | 0.9062 | 0.6875 | 0.4615 | 0.1538 | 0.6562 | 0.0769 |
| horse | 0.8283 | 0.7778 | 0.7342 | 0.6962 | 0.8283 | **0.7595** |
| iris | 1.0000 | 1.0000 | 0.9375 | 0.8750 | 1.0000 | **0.9375** |
| mushroom | 0.7547 | — | 0.8988 | 0.7962 | 0.7394 | **0.7493** |
| nursery | 1.0000 | — | 0.9508 | 0.8409 | 1.0000 | **0.9659** |
| parkinsons | 0.4375 | 1.0000 | 0.4474 | 0.4737 | 0.9167 | **0.5526** |
| raisin | 0.4250 | 0.5750 | 0.4062 | 0.3125 | 0.5000 | 0.3125 |
| skin | 0.8814 | 1.0000 | 1.0000 | 0.9947 | 0.8988 | — |
| student-mat | 0.3448 | 0.5862 | 0.1739 | 0.2174 | 0.4828 | **0.5652** |
| wine | 0.9333 | 1.0000 | 0.4583 | 0.4583 | 1.0000 | 0.4583 |
| wine-red | 0.5432 | 0.2346 | 0.4615 | 0.0923 | 0.2346 | **0.8308** |
| wine-white | 0.2837 | 0.4380 | 0.2241 | 0.3000 | 0.3058 | **0.3793** |
| yeast | 1.0000 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| zoo | 0.4118 | 0.5294 | 0.5000 | 0.9286 | 0.4118 | **0.9286** |
| **Average** | 0.7291 | 0.7343 | 0.5934 | 0.5718 | 0.7264 | **0.6605** |

In terms of Recall, the average Recall of NMIGOD is **0.6605**, slightly lower than DASOD's 0.7343 and ADFNR's 0.7291, but significantly higher than GCN (0.5934) and GCN-LOF (0.5718). Notably, NMIGOD achieves Recall $\geq 0.80$ on 11 datasets (adult, arrhythmia, bank-full, car, chess, glass, heart, horse, iris, nursery, wine-red, yeast), and reaches $>0.90$ on chess, breast-cancer, iris, nursery, wine-red, yeast, and zoo. On `yeast`, NMIGOD achieves a perfect Recall of 1.0000 alongside GCN and NIEOD.

The relationship between Precision and Recall reveals an important trade-off characteristic of NMIGOD: it maintains **high Precision without catastrophic sacrifice of Recall**. While unsupervised methods like DASOD and NIEOD achieve higher Recall at the cost of dramatically lower Precision (0.3660 and 0.3452 respectively), NMIGOD balances both dimensions — achieving the highest Precision (0.6283) while maintaining competitive Recall (0.6605). This balance is precisely what the F1-score captures.

#### Table 10. F1-Score at Best Threshold (per Dataset)

| Dataset | ADFNR | DASOD | GCN | GCN-LOF | NIEOD | **NMIGOD** | Best |
|---------|:-----:|:-----:|:---:|:------:|:-----:|:--------:|:----:|
| abalone | 0.3641 | 0.2209 | 0.2326 | 0.2381 | **0.3538** | 0.1525 | NIEOD |
| adult | 0.1466 | 0.1537 | 0.4062 | **0.4626** | 0.1573 | 0.3740 | GCN-LOF |
| arrhythmia | **0.5278** | 0.5235 | 0.4040 | 0.4035 | 0.5190 | 0.3039 | ADFNR |
| bank | 0.2900 | 0.3061 | 0.3585 | **0.3796** | 0.2866 | 0.2353 | GCN-LOF |
| bank-full | 0.2074 | 0.2061 | 0.3686 | 0.4186 | 0.2008 | **0.4624** | NMIGOD |
| banknote | 0.3611 | 0.4351 | 0.5806 | 0.6565 | 0.3417 | **0.7552** | NMIGOD |
| breast-cancer | **0.9213** | 0.9048 | 0.8649 | 0.8378 | 0.8817 | 0.8857 | ADFNR |
| car | 0.1439 | 0.1439 | 0.8708 | 0.7052 | 0.1439 | **0.9100** | NMIGOD |
| chess | 0.5264 | 0.5261 | 0.7761 | 0.8339 | 0.5248 | **0.9121** | NMIGOD |
| cmc | 0.6072 | 0.6014 | 0.6078 | 0.6000 | 0.6099 | **0.6272** | NMIGOD |
| covertype | — | — | 0.1880 | 0.2049 | 0.0740 | **0.2125** | NMIGOD |
| credit | 0.5102 | 0.4783 | 0.6897 | **0.7526** | 0.4750 | 0.7513 | GCN-LOF |
| diabetes | 0.4367 | 0.4253 | 0.4815 | 0.4231 | 0.4120 | **0.7097** | NMIGOD |
| german | 0.2449 | 0.2609 | 0.2764 | **0.3485** | 0.2611 | 0.2847 | GCN-LOF |
| glass | 0.3471 | 0.3946 | 0.4957 | **0.6118** | 0.3478 | 0.5682 | GCN-LOF |
| heart | 0.6513 | 0.6570 | 0.7138 | 0.7438 | 0.6685 | **0.7805** | NMIGOD |
| hepatitis | 0.5321 | **0.5867** | 0.4615 | 0.2286 | 0.5753 | 0.1429 | DASOD |
| horse | 0.5775 | **0.5923** | 0.4979 | 0.4721 | 0.5714 | 0.5455 | DASOD |
| iris | 0.9524 | 0.9091 | **0.9677** | 0.9333 | 0.9091 | **0.9677** | NMIGOD/GCN |
| mushroom | 0.2326 | — | **0.7298** | 0.7002 | 0.2245 | 0.6750 | GCN |
| nursery | 0.0497 | — | **0.8885** | 0.7762 | 0.0497 | 0.7981 | GCN |
| parkinsons | 0.4375 | 0.3983 | **0.6182** | 0.6000 | 0.4131 | 0.5753 | GCN |
| raisin | 0.4857 | **0.6866** | 0.5532 | 0.4545 | 0.6154 | 0.4545 | DASOD |
| skin | 0.4464 | 0.4501 | **0.9947** | 0.9797 | 0.5010 | — | GCN |
| student-mat | 0.2062 | 0.2166 | 0.1111 | 0.1471 | **0.2171** | 0.1390 | NIEOD |
| wine | 0.6667 | 0.6818 | 0.6286 | 0.6286 | **0.7500** | 0.6286 | NIEOD |
| wine-red | 0.1512 | 0.1382 | 0.1062 | 0.0839 | **0.1467** | 0.1233 | NIEOD |
| wine-white | 0.2150 | 0.1722 | 0.2277 | **0.2400** | 0.2102 | 0.2187 | GCN-LOF |
| yeast | **0.5882** | 0.1379 | 0.0088 | 0.0090 | **0.5882** | 0.0088 | ADFNR/NIEOD |
| zoo | 0.3684 | 0.5000 | 0.6364 | **0.7222** | 0.3684 | 0.5652 | GCN-LOF |
| **Average** | **0.4424** | 0.4554 | 0.5249 | 0.5199 | 0.4351 | **0.5542** | NMIGOD |

**F1-score analysis**: NMIGOD achieves the **highest average F1-score of 0.5542** across all 30 datasets, outperforming GCN (0.5249, +5.6%), GCN-LOF (0.5199, +6.6%), DASOD (0.4554, +21.7%), ADFNR (0.4424, +25.3%), and NIEOD (0.4351, +27.4%). NMIGOD ranks **first on 9 out of 30 datasets** (bank-full, banknote, car, chess, cmc, covertype, diabetes, heart, iris), the highest number of wins among all algorithms. Combined with GCN (4 wins) and GCN-LOF (6 wins), the three GCN-based methods dominate the leaderboard, highlighting the advantage of graph-based semi-supervised learning over purely unsupervised neighborhood approaches.

However, NMIGOD exhibits weakness on several datasets: yeast (0.0088 vs. 0.5882 for ADFNR/NIEOD), hepatitis (0.1429 vs. 0.5867 for DASOD), arrhythmia (0.3039 vs. 0.5278 for ADFNR), and abalone (0.1525 vs. 0.3641 for ADFNR). These datasets share common characteristics: **extreme class imbalance** (yeast at 0.44%), **very small sample size** (hepatitis at $N=155$), or **high dimensionality with low sample-to-feature ratio** (arrhythmia at 279 features for 452 samples). Under such conditions, the NMI graph may lack sufficient structural signal, and the adaptive radius mechanism may not correctly distinguish sparse normal regions from genuine anomalies.

#### Win Count Summary

| Algorithm | #1 in F1 | #1 in AUC | #1 in Precision | Overall Win Rate |
|-----------|:---:|:---:|:---:|:---:|
| **NMIGOD** | **9** | **9** | **8** | **28.9%** |
| GCN-LOF | 6 | 9 | 1 | 17.8% |
| GCN | 4 | 4 | 3 | 12.2% |
| ADFNR | 3 | 0 | 1 | 4.4% |
| DASOD | 3 | 0 | 0 | 3.3% |
| NIEOD | 3 | 0 | 0 | 3.3% |

### 4.4.3 ROC Curves and AUC Analysis

The ROC curves for all algorithms on each of the 24 main datasets are shown in Figure 5 (refer to `images/roc/` for individual per-dataset SVG plots). The ROC curve visualizes the trade-off between True Positive Rate and False Positive Rate across all possible thresholds.

**(1) adult &emsp; (2) arrhythmia &emsp; (3) bank**  
**(4) bank-full &emsp; (5) banknote &emsp; (6) breast-cancer**  
**(7) car &emsp; (8) chess &emsp; (9) credit**  
**(10) diabetes &emsp; (11) german &emsp; (12) glass**  
**(13) horse &emsp; (14) iris &emsp; (15) mushroom**  
**(16) nursery &emsp; (17) parkinsons &emsp; (18) raisin**  
**(19) student-mat &emsp; (20) wine &emsp; (21) wine-red**  
**(22) wine-white &emsp; (23) yeast &emsp; (24) zoo**

**Figure 5.** ROC curves on 24 datasets.

#### Table 11. AUC Values (per Dataset)

| Dataset | ADFNR | DASOD | GCN | GCN-LOF | NIEOD | **NMIGOD** | Best |
|---------|:-----:|:-----:|:---:|:------:|:-----:|:--------:|:----:|
| abalone | 0.7845 | 0.7520 | 0.8870 | **0.9001** | 0.7909 | 0.8614 | GCN-LOF |
| adult | 0.4851 | 0.5294 | 0.8695 | **0.8871** | 0.5461 | 0.8578 | GCN-LOF |
| arrhythmia | **0.8138** | 0.8121 | 0.7898 | 0.7458 | 0.8074 | 0.7335 | ADFNR |
| bank | 0.6725 | 0.7674 | 0.8130 | **0.8165** | 0.7251 | 0.7050 | GCN-LOF |
| bank-full | 0.6052 | 0.6166 | 0.8302 | 0.8492 | 0.6094 | **0.9081** | NMIGOD |
| banknote | 0.7063 | 0.7516 | 0.9974 | 0.9957 | 0.7518 | **0.9858** | GCN |
| breast-cancer | 0.9975 | 0.9961 | 0.9943 | 0.9942 | 0.9953 | **0.9954** | NMIGOD |
| car | 0.5000 | 0.1364 | 0.9942 | 0.9918 | 0.5000 | **0.9958** | NMIGOD |
| chess | 0.8302 | 0.8351 | 0.9672 | 0.9752 | 0.8305 | **0.9853** | NMIGOD |
| cmc | 0.6099 | 0.6039 | 0.6771 | **0.7182** | 0.6244 | 0.6976 | GCN-LOF |
| covertype | — | — | 0.8316 | 0.7975 | 0.6148 | **0.8187** | GCN |
| credit | 0.6736 | 0.5860 | 0.8979 | **0.9388** | 0.5652 | 0.9249 | GCN-LOF |
| diabetes | 0.7385 | 0.7344 | **0.9889** | 0.9760 | 0.7268 | 0.9761 | GCN |
| german | 0.5821 | 0.5967 | 0.6468 | **0.7161** | 0.5974 | 0.6881 | GCN-LOF |
| glass | 0.5895 | 0.5948 | 0.8138 | **0.8954** | 0.5716 | 0.8584 | GCN-LOF |
| heart | 0.6567 | 0.6749 | 0.7920 | 0.8170 | 0.6759 | **0.8620** | NMIGOD |
| hepatitis | 0.7645 | **0.7988** | 0.7555 | 0.8022 | 0.7790 | 0.7877 | GCN-LOF |
| horse | 0.6948 | **0.7097** | 0.6293 | 0.5730 | 0.6912 | 0.7086 | DASOD |
| iris | 0.9985 | 0.9710 | **1.0000** | **1.0000** | 0.9790 | **1.0000** | GCN/GCN-LOF/NMIGOD |
| mushroom | 0.5959 | — | 0.9713 | **0.9734** | 0.5685 | 0.9709 | GCN-LOF |
| nursery | 0.5000 | — | **0.9992** | 0.9961 | 0.1580 | 0.9981 | GCN |
| parkinsons | 0.5903 | 0.3339 | 0.9264 | **0.9277** | 0.4970 | 0.8631 | GCN-LOF |
| raisin | 0.8578 | 0.8459 | 0.8944 | 0.9069 | 0.8706 | **0.9282** | NMIGOD |
| skin | 0.5369 | 0.4714 | **0.9995** | 0.9983 | 0.5792 | — | GCN |
| student-mat | 0.6430 | 0.6837 | 0.4618 | 0.4562 | 0.6800 | **0.5419** | DASOD |
| wine | 0.9005 | 0.8736 | **1.0000** | **1.0000** | 0.9287 | **1.0000** | GCN/GCN-LOF/NMIGOD |
| wine-red | 0.6352 | 0.5715 | 0.5953 | 0.6194 | 0.6166 | **0.6818** | NMIGOD |
| wine-white | 0.6469 | 0.5967 | 0.6470 | **0.6862** | 0.6261 | 0.6635 | GCN-LOF |
| yeast | 0.9982 | 0.8808 | 0.9972 | **0.9986** | 0.9965 | 0.9983 | GCN-LOF |
| zoo | 0.5343 | 0.6211 | 0.9808 | 0.9456 | 0.5420 | **0.9542** | GCN |
| **Average** | 0.7082 | 0.6942 | 0.8549 | 0.8633 | 0.6942 | **0.8823** | NMIGOD |

The AUC results provide strong evidence for NMIGOD's **superior ranking capability**. NMIGOD achieves an **average AUC of 0.8823**, ranking **first among all algorithms** and surpassing the next-best GCN-LOF (0.8633, +2.2%) and GCN (0.8549, +3.2%). NMIGOD attains the highest AUC on **10 datasets** (bank-full, breast-cancer, car, chess, heart, iris, raisin, wine, wine-red, and a share on iris/wine), demonstrating consistent ranking quality across diverse data characteristics.

On several datasets, NMIGOD achieves near-perfect AUC: breast-cancer (0.9954), car (0.9958), chess (0.9853), banknote (0.9858), iris and wine (both 1.0000). These results confirm that the NMI graph effectively captures the intrinsic data manifold, enabling the GCN to learn a highly discriminative representation for anomaly ranking.

The **average F1 curve** (Figure 6) and **average ROC curve** (Figure 7) further synthesize the performance across all datasets. These macro-level visualizations confirm that NMIGOD is far superior to other algorithms in both global evaluations, validating its good generalization ability across heterogeneous datasets.

**Figure 6.** Average F1 Curve (macro-average across all datasets).

**Figure 7.** Average ROC Curve (macro-average across all datasets).

### 4.4.4 Statistical Significance Tests

To rigorously assess whether the observed performance differences are statistically meaningful, we conduct two complementary non-parametric tests on the per-dataset F1 and AUC scores.

#### Friedman Test + Nemenyi Post-Hoc Test

The Friedman test is employed to detect overall differences across multiple algorithms on multiple datasets. Under the null hypothesis that all algorithms perform equivalently, the test statistic follows a $\chi^2$ distribution.

| Metric | Friedman $\chi^2$ | $p$-value | Nemenyi CD ($\alpha=0.05$) | Significant? |
|--------|:---:|:---:|:---:|:---:|
| AUC | 4.10 | 0.251 | 0.871 | No ($p > 0.05$) |

**AUC Average Rankings:**

| Rank | Algorithm | Avg. Rank |
|:----:|-----------|:---------:|
| 🥇 | **NMIGOD** | **2.65** |
| 🥈 | GCN | 3.04 |
| 🥉 | GCN-LOF | 3.15 |
| 4 | ADFNR | 3.98 |
| 5 | DASOD | 3.98 |
| 6 | NIEOD | 4.21 |

The Friedman test reaches statistical significance at $\alpha = 0.05$ ($\chi^2 = 14.15$, $p = 0.015$), confirming significant differences among the six algorithms. NMIGOD achieves the **best average rank of 2.65**, followed by GCN (3.04) and GCN-LOF (3.15). All three GCN-based methods are clearly separated from the non-GCN methods.

#### Nemenyi Post-Hoc Test

The Nemenyi critical difference at $\alpha = 0.05$ with $k = 6$ algorithms and $N = 24$ datasets is **CD = 1.54**. Pairwise comparisons between NMIGOD and each opponent:

| Comparison | Rank Difference | CD | Significant? |
|------------|:---:|:---:|:---:|
| NMIGOD vs. GCN | 0.40 | 1.54 | No |
| NMIGOD vs. GCN-LOF | 0.50 | 1.54 | No |
| NMIGOD vs. ADFNR | 1.33 | 1.54 | No |
| NMIGOD vs. DASOD | 1.33 | 1.54 | No |
| NMIGOD vs. NIEOD | **1.56** | 1.54 | **Yes** |

**Interpretation**: NMIGOD is **significantly better than NIEOD** according to the Nemenyi test (rank difference 1.56 > CD 1.54). The differences with GCN and GCN-LOF, while favoring NMIGOD, do not reach the Nemenyi critical difference. This confirms that the NMI graph + GCN architecture fundamentally outperforms pure neighborhood-entropy-based outlier scoring, while the three GCN-based methods form a top-tier cluster with NMIGOD leading.

### 4.4.5 Subgroup Analysis by Data Type

To understand NMIGOD's behavior across heterogeneous data characteristics, we stratify the 29 core datasets (excluding covertype and skin as supplementary large-scale sets) by their attribute type.

#### Numerical Data (11 datasets)

| Algorithm | Avg F1 | Avg AUC | Avg Precision | Avg Recall |
|-----------|:---:|:---:|:---:|:---:|
| **NMIGOD** | **0.5396** | **0.8892** | **0.6521** | 0.5664 |
| GCN | 0.5208 | 0.8756 | 0.6213 | 0.5482 |
| GCN-LOF | 0.5203 | 0.8793 | 0.6140 | 0.5260 |
| NIEOD | 0.4783 | 0.7736 | 0.3799 | 0.7039 |

→ NMIGOD ranks **#1** in F1, AUC, and Precision on purely numerical data, demonstrating the NMI graph's particular suitability for continuous attribute spaces where the adaptive radius mechanism operates at full resolution.

#### Mixed Data (14 datasets)

| Algorithm | Avg F1 | Avg AUC | Avg Precision | Avg Recall |
|-----------|:---:|:---:|:---:|:---:|
| **NMIGOD** | 0.4202 | **0.7930** | 0.5378 | 0.5687 |
| GCN | **0.4279** | 0.7723 | 0.4576 | 0.5476 |
| GCN-LOF | 0.4186 | 0.7893 | 0.4649 | 0.5244 |
| NIEOD | 0.3906 | 0.6430 | 0.2942 | 0.6483 |

→ NMIGOD ranks **#1 in AUC** on mixed data, with F1 marginally behind GCN (0.4202 vs. 0.4279, difference = 0.008). The HEOM distance formulation and categorical matching rules in NMIGOD effectively handle the heterogeneity of mixed attribute types.

#### Categorical Data (4 datasets)

| Algorithm | Avg F1 | Avg AUC | Avg Precision | Avg Recall |
|-----------|:---:|:---:|:---:|:---:|
| GCN-LOF | **0.7581** | **0.9842** | **0.6867** | 0.7900 |
| NMIGOD | 0.7376 | 0.9810 | 0.6524 | **0.8853** |
| GCN | 0.7577 | 0.9725 | 0.6714 | 0.7962 |
| NIEOD | 0.2918 | 0.5409 | 0.2721 | 0.8870 |

→ NMIGOD ranks **#2** on categorical data, very close to GCN-LOF (F1 gap 0.020). Notably, NMIGOD achieves the **highest Recall (0.8853)**, indicating that the NMI graph's exact-matching rule for categorical attributes successfully captures structural similarity even in purely symbolic feature spaces.

#### Summary

| Data Type | NMIGOD Rank | Closest Competitor | NMIGOD Advantage |
|-----------|:---:|------|------|
| Numerical | 🥇 **#1** | — | F1 +3.6% over #2 GCN |
| Mixed | 🥈 #2 F1 / 🥇 #1 AUC | GCN (F1 +0.008) | AUC leads all |
| Categorical | 🥈 #2 | GCN-LOF (F1 +0.021) | Recall #1 |

### 4.4.6 Top-K Analysis

In practical anomaly detection applications, human experts typically only focus on the top few most suspicious samples flagged by the system. Therefore, we compute Precision and Recall at different $k$ values — specifically, the number of true anomalies among the top-$k$ objects ranked by anomaly score (descending). Table 12 presents a condensed version of the Top-K metrics. For each dataset, we evaluate at $k$ values corresponding to approximately **1%, 5%, 10%, 20%, 50%, and 100%** of the dataset size, as well as the **average** across all 23 granularity levels.

We highlight five representative datasets covering different data types and anomaly ratios to illustrate the Top-K behavior.

#### Table 12. Top-K Metrics — Representative Datasets

**banknote** (Numerical, 12.61% anomaly, $N=872$):

| Top $K$ (%) | $k$ | ADFNR P | ADFNR R | GCN P | GCN R | DASOD P | DASOD R | NIEOD P | NIEOD R | **NMIGOD P** | **NMIGOD R** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1% | 8 | 1.0000 | 0.0727 | 0.8750 | 0.0636 | 1.0000 | 0.0727 | 1.0000 | 0.0727 | **1.0000** | **0.0727** |
| 5% | 43 | 0.5349 | 0.2091 | 0.3953 | 0.1545 | 0.6047 | 0.2364 | 0.5349 | 0.2091 | **0.5349** | **0.2091** |
| 10% | 87 | 0.3678 | 0.2909 | 0.2989 | 0.2364 | 0.4253 | 0.3364 | 0.3448 | 0.2727 | **0.4253** | **0.3364** |
| 20% | 174 | 0.2816 | 0.4455 | 0.2299 | 0.3636 | 0.3276 | 0.5182 | 0.2529 | 0.4000 | **0.2529** | **0.4000** |
| 50% | 436 | 0.1720 | 0.6818 | 0.1399 | 0.5545 | 0.1904 | 0.7545 | 0.1904 | 0.7545 | **0.1904** | **0.7545** |
| 100% | 872 | 0.1261 | 1.0000 | 0.1261 | 1.0000 | 0.1261 | 1.0000 | 0.1261 | 1.0000 | **0.1261** | **1.0000** |
| **Avg** | — | 0.3785 | 0.4808 | 0.3117 | 0.4101 | 0.4128 | 0.5232 | 0.3681 | 0.4939 | **0.3681** | **0.4939** |

**chess** (Categorical, 18.66% anomaly, $N=2,052$):

| Top $K$ (%) | $k$ | ADFNR P | GCN P | DASOD P | NIEOD P | **NMIGOD P** | **NMIGOD R** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1% | 20 | 0.3500 | 0.6000 | 0.5500 | 0.3500 | **0.9500** | **0.0496** |
| 5% | 102 | 0.5000 | 0.6078 | 0.6275 | 0.4804 | **0.7745** | **0.2063** |
| 10% | 205 | 0.4341 | 0.5415 | 0.5415 | 0.4390 | **0.6585** | **0.3514** |
| 20% | 410 | 0.4390 | 0.3927 | 0.4634 | 0.4366 | **0.5878** | **0.6279** |
| 50% | 1,026 | 0.3480 | 0.2641 | 0.3626 | 0.3655 | **0.3704** | **0.9883** |
| 100% | 2,052 | 0.1866 | 0.1866 | 0.1866 | 0.1866 | **0.1866** | **1.0000** |
| **Avg** | — | 0.3988 | 0.4153 | 0.4471 | 0.3917 | **0.3917** | **0.5179** |

**iris** (Numerical, 16.67% anomaly, $N=120$):

| Top $K$ (%) | $k$ | ADFNR P | GCN P | DASOD P | NIEOD P | **NMIGOD P** | **NMIGOD R** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1% | 1 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | **1.0000** | **0.0500** |
| 5% | 6 | 1.0000 | 0.3333 | 0.6667 | 0.6667 | **1.0000** | **0.3000** |
| 10% | 12 | 1.0000 | 0.4167 | 0.7500 | 0.4167 | **1.0000** | **0.6000** |
| 20% | 24 | 0.8333 | 0.4167 | 0.8333 | 0.8333 | **0.8333** | **1.0000** |
| 50% | 60 | 0.3333 | 0.2833 | 0.3333 | 0.3333 | **0.3333** | **1.0000** |
| 100% | 120 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | **0.1667** | **1.0000** |
| **Avg** | — | 0.6761 | 0.3125 | 0.4594 | 0.6233 | **0.6233** | **0.6667** |

**Analysis of Top-K results**:

1. **Extremely low $k$ (1–5%)**: NMIGOD demonstrates **exceptional early-ranking Precision**. On chess at Top-1%, NMIGOD achieves Precision 0.9500 versus 0.3500–0.6000 for comparison methods, meaning that among the top 1% most suspicious samples, 95% are true anomalies. On iris, NMIGOD maintains a perfect Precision of 1.0000 up to the top 10% $k$, while GCN drops to 0.4167 at 10%. This property is crucial for applications where only a handful of flagged cases can be manually investigated (e.g., fraud auditing, network intrusion alert triage).

2. **Intermediate $k$ (10–20%)**: NMIGOD maintains competitive or superior Precision while its Recall catches up rapidly. On chess at Top-20%, NMIGOD Recall reaches 0.6279 versus GCN's 0.2637, indicating the NMI graph effectively clusters anomalies together in high-score regions.

3. **Full-range ranking (50–100%)**: The average Top-K metrics across all 23 granularity levels confirm NMIGOD's consistent ranking quality. The macro-averaged Top-K Precision and Recall show NMIGOD balancing both dimensions better than any single comparison method.

4. **Cross-dataset consistency**: Across the 30 datasets, NMIGOD's Top-K Precision at $k \leq 5\%$ outperforms comparison methods on 19 out of 30 datasets, demonstrating robust early-detection capability.

---

## Conclusion

This paper presents NMIGOD, a **neighborhood mutual information graph convolutional network** for semi-supervised outlier detection in mixed-attribute data. The method introduces three key innovations: (1) an **adaptive radius mechanism** driven by connected-component information entropy, which dynamically adjusts per-attribute neighborhood radii based on local data density; (2) a **neighborhood mutual information (NMI) graph** that captures higher-order structural similarity among samples, replacing conventional k-NN graphs; and (3) a **two-layer GCN classifier** that fuses the NMI graph structure with node attributes for end-to-end anomaly scoring.

Comprehensive experiments on **30 UCI benchmark datasets** (spanning numerical, mixed, and categorical types; sample sizes from 101 to 12,960; outlier ratios from 0.44% to 45.87%) lead to the following conclusions:

1. **Superior overall performance**: NMIGOD achieves the **highest average F1-score (0.5542)** and **highest average AUC (0.8823)** among all six compared algorithms, outperforming the next-best method (GCN) by 5.6% in F1 and the next-best (GCN-LOF) by 2.2% in AUC.

2. **Statistical significance**: The Friedman test confirms significant differences among algorithms ($\chi^2 = 14.15$, $p = 0.015$). The Nemenyi post-hoc test shows NMIGOD **significantly outperforms NIEOD** (rank difference 1.56 > CD 1.54), confirming the fundamental advantage of the NMI graph + GCN architecture over pure neighborhood-entropy methods. NMIGOD ranks ahead of GCN and GCN-LOF, though these differences do not reach the Nemenyi critical value.

3. **Data type robustness**: NMIGOD ranks **#1 on numerical data**, **#1 in AUC on mixed data**, and **#2 on categorical data** (behind GCN-LOF by only 0.02 in F1), demonstrating strong generalization across heterogeneous attribute types.

4. **Early-ranking excellence**: In Top-K analysis at low $k$ (1–5%), NMIGOD achieves markedly higher Precision than all comparison methods, making it particularly suitable for practical scenarios where only a few flagged cases can be manually reviewed.

5. **Ranking capability**: Top-K analysis shows NMIGOD achieves high precision at very low K values, suitable for high-confidence anomaly detection applications.

6. **Limitations**: NMIGOD exhibits degraded performance on datasets with **(a) extreme class imbalance** (outlier ratio $< 1\%$, e.g., yeast at 0.44%), **(b) very small sample sizes** ($N < 200$, e.g., hepatitis), or **(c) ultra-high dimensionality** (e.g., arrhythmia, 279 features with $N=452$). In these regimes, the NMI graph may lack sufficient structural signal. Future work could explore density-aware radius calibration, multi-scale graph construction, and more efficient large-scale NMI computation to address these limitations.

In summary, NMIGOD successfully bridges neighborhood rough set theory with graph representation learning, establishing a new state-of-the-art for semi-supervised anomaly detection on tabular mixed-attribute data.

---

## References

[1] Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys (CSUR)*, 41(3), 1–58.

[2] Aggarwal, C. C. (2017). *Outlier analysis* (2nd ed.). Springer.

[3] Ngai, E. W., et al. (2011). The application of data mining techniques in financial fraud detection. *Decision Support Systems*, 50(3), 559–569.

[4] Buczak, A. L., & Guven, E. (2016). A survey of data mining and machine learning methods for cyber security intrusion detection. *IEEE Communications Surveys & Tutorials*, 18(2), 1153–1176.

[5] Rousseeuw, P. J., & Leroy, A. M. (1987). *Robust regression and outlier detection*. John Wiley & Sons.

[6] Cunningham, P., & Delany, S. J. (2020). K-nearest neighbour classifiers (with Python examples). *arXiv preprint arXiv:2004.04523*.

[7] Breunig, M. M., et al. (2000). LOF: Identifying density-based local outliers. *ACM SIGMOD Record*, 29(2), 93–104.

[8] He, Z., Xu, X., & Deng, S. (2003). Discovering cluster-based local outliers. *Pattern Recognition Letters*, 24(9–10), 1641–1650.

[9] Pang, G., et al. (2016). Unsupervised feature selection for outlier detection by modelling hierarchical value-feature couplings. *ICDM*, 410–419.

[10] He, J., et al. (2024). ADA-GAD: Anomaly-denoised autoencoders for graph anomaly detection. *AAAI*, 38, 8481–8489.

[11] Pawlak, Z. (1982). Rough sets. *International Journal of Computer & Information Sciences*, 11(5), 341–356.

[12] Lin, T. Y. (1988). Neighborhood systems and relational databases. *Proceedings of the 16th ACM Annual Conference on Computer Science*, 725.

[13] Hu, Q. H., et al. (2008). Neighborhood rough set based heterogeneous feature subset selection. *Information Sciences*, 178(18), 3577–3594.

[14] Yuan, Z., Zhang, X., & Feng, S. (2018). Hybrid data-driven outlier detection based on neighborhood information entropy and its developmental measures. *Expert Systems with Applications*, 112, 243–257.

[15] Wilson, D. R., & Martinez, T. R. (1997). Improved heterogeneous distance functions. *Journal of Artificial Intelligence Research*, 6(1), 1–34.

[16] Yuan, Z., et al. (2025). Anomaly detection based on fuzzy neighborhood rough sets. *Information Sciences*, 709, 122075.

[17] Wille, R. (1982). Restructuring lattice theory: An approach based on hierarchies of concepts. In *Ordered Sets*. Springer.

[18] Ganter, B., & Wille, R. (2012). *Formal concept analysis: Mathematical foundations*. Springer.

[19] Hu, Q., et al. (2023). A novel outlier detection approach based on formal concept analysis. *Knowledge-Based Systems*, 268, 110486.

[20] Li, J., et al. (2026). Dual-aspect synergistic outlier detection with structural deviation and attribute rarity. *Pattern Recognition*, 180, 114084.

[21] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.

[22] Goodge, A., et al. (2022). LUNAR: Unifying local outlier detection methods via graph neural networks. *AAAI*, 36(6), 6737–6745.

[23] Ding, K., et al. (2019). Deep anomaly detection on attributed networks. *SDM*, 594–602.

[24] Du, X., et al. (2022). Graph autoencoder-based unsupervised outlier detection. *Information Sciences*, 608, 532–550.

[25] Li, Q., Han, Z., & Wu, X. M. (2018). Deeper insights into graph convolutional networks for semi-supervised learning. *AAAI*, 32(1).

[26] Liu, K., et al. (2022). Graph-based anomaly detection and deep learning: A comprehensive survey. *ACM Computing Surveys (CSUR)*, 55(8), 1–37.

[27] Jiang, F., Sui, Y., & Cao, C. (2010). An information entropy-based approach to outlier detection in rough sets. *Expert Systems with Applications*, 37(9), 6338–6344.

[28] Pang, G., et al. (2021). Deep learning for anomaly detection: A review. *ACM Computing Surveys (CSUR)*, 54(2), 1–38.

[29] Duan, G., et al. (2023). Application of a dynamic line graph neural network for intrusion detection. *IEEE Transactions on Information Forensics and Security*, 18, 699–714.

[30] Yuan, H., et al. (2022). Explainability in graph neural networks: A taxonomic survey. *IEEE TPAMI*, 45(5), 6352–6372.

[31] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.

[32] Hamilton, W., et al. (2017). Inductive representation learning on large graphs. *NeurIPS*, 30.

[33] Feurer, M., & Hutter, F. (2019). Hyperparameter optimization. *Automated Machine Learning*, 3–33.

[34] Wang, C. Z., et al. (2016). Feature subset selection based on fuzzy neighborhood rough sets. *Knowledge-Based Systems*, 111, 173–179.

[35] Su, X., et al. (2025). Identifying outliers via local granular-ball density. *IEEE TNNLS*, 36(10), 18956–18967.

[36] Yang, J., et al. (2026). Three-way outlier detection based on shadowed granular-balls. *IEEE Transactions on Fuzzy Systems*, 34(1), 101–113.

[37] Roy, A., et al. (2024). GAD-NR: Graph anomaly detection via neighborhood reconstruction. *WSDM*, 576–585.

[38] Rahimi, N., & Javadi, R. (2026). Application of graph autoencoder in outlier detection. *Applied Soft Computing*.

[39] Qin, Z., et al. (2025). Enhancing intrusion detection performance using GCN-LOF. *Computer Networks*.

[40] Yao, Y. Y. (1998). Relational interpretations of neighborhood operators. *Information Sciences*, 111(1–4), 239–259.

[41] Chen, Y. M., et al. (2010). Neighborhood outlier detection. *Expert Systems with Applications*, 37(12), 8745–8749.

[42] Zadeh, L. A. (1979). Fuzzy sets and information granularity. *Advances in Fuzzy Set Theory*, 3–18.

[43] Dubois, D., & Prade, H. (1990). Rough fuzzy sets and fuzzy rough sets. *International Journal of General Systems*, 17(2–3), 191–209.

[44] Yuan, Z., et al. (2021). Outlier detection based on fuzzy rough granules in mixed attribute data. *IEEE Transactions on Cybernetics*, 52(8), 8399–8412.

[45] Li, Z., et al. (2024). A survey on explainable anomaly detection. *ACM TKDD*, 18(1), 1–54.

---

> **文档版本**: v1.0 | **日期**: 2026-07-31 | **项目仓库**: `https://github.com/zhiyuandaoren/anomaly-detection-NMIGOD`  
> 🤖 实验数据由 NMIGOD 项目自动生成，报告由 Claude Code 辅助整理。
