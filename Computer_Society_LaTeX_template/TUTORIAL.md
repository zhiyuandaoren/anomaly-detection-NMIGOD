# IEEE Computer Society LaTeX 模板使用教程

> 适用于 `Computer_Society_LaTeX_template/` 目录下的 IEEEtran 模板  
> 模板版本：IEEEtran v1.8b | 适用场景：IEEE 期刊/会议论文投稿

---

## 目录

1. [模板文件说明](#1-模板文件说明)
2. [快速开始](#2-快速开始)
3. [文档类选项](#3-文档类选项)
4. [前置信息（Front Matter）](#4-前置信息front-matter)
5. [正文结构](#5-正文结构)
6. [图表插入](#6-图表插入)
7. [数学公式](#7-数学公式)
8. [算法伪代码](#8-算法伪代码)
9. [参考文献](#9-参考文献)
10. [作者传记](#10-作者传记)
11. [编译与排错](#11-编译与排错)
12. [适配 NMIGOD 论文指南](#12-适配-nmigod-论文指南)

---

## 1. 模板文件说明

模板目录 `Computer_Society_LaTeX_template/` 包含以下文件：

| 文件 | 说明 |
|------|------|
| `bare_jrnl_new_sample4.tex` | **主示例文件** — 完整展示期刊论文所有元素 |
| `bare_jrnl_new_sample4.pdf` | 示例文件的编译结果 |
| `New_IEEEtran_how-to.tex` | **使用指南** — IEEE 官方简化教程 |
| `New_IEEEtran_how-to.pdf` | 使用指南的编译结果 |
| `fig1.png` | 示例图片（用于演示插图功能） |

---

## 2. 快速开始

### 2.1 复制模板

```bash
cp bare_jrnl_new_sample4.tex my_paper.tex
```

### 2.2 最基本的文档框架

```latex
\documentclass[lettersize,journal]{IEEEtran}  % 期刊模式

% ===== 必须加载的宏包 =====
\usepackage{amsmath,amsfonts}       % 数学公式
\usepackage{algorithmic}            % 伪代码
\usepackage{algorithm}              % 算法浮动体
\usepackage{array}                  % 表格增强
\usepackage{graphicx}               % 插入图片
\usepackage{cite}                   % 引用
\usepackage{url}                    % URL 链接
\usepackage{verbatim}               % 代码块
\usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{subfig}  % 子图

\begin{document}

\title{Your Paper Title}

\author{Author Name,~\IEEEmembership{Member,~IEEE,}
\thanks{Corresponding author email.}}

\maketitle

\begin{abstract}
Your abstract here.
\end{abstract}

\begin{IEEEkeywords}
Keyword1, Keyword2, Keyword3.
\end{IEEEkeywords}

\section{Introduction}
Your content here.

\begin{thebibliography}{1}
\bibitem{ref1} Reference here.
\end{thebibliography}

\end{document}
```

### 2.3 编译器要求

- **推荐**: PDFLaTeX（编译 `.tex` → `.pdf`）
- **备用**: XeLaTeX / LuaLaTeX（如果需要中文支持）
- **文献**: BibTeX（可选，也可手动写 `thebibliography`）

```bash
# 命令行编译（推荐两次以获得正确的交叉引用）
pdflatex my_paper.tex
pdflatex my_paper.tex
```

---

## 3. 文档类选项

`\documentclass` 的第一个参数决定论文类型：

### 3.1 常用模式

| 命令 | 用途 |
|------|------|
| `\documentclass[journal]{IEEEtran}` | **期刊论文**（默认 10pt, 双栏） |
| `\documentclass[conference]{IEEEtran}` | 会议论文 |
| `\documentclass[10pt,journal,compsoc]{IEEEtran}` | **Computer Society 期刊** |
| `\documentclass[conference,compsoc]{IEEEtran}` | Computer Society 会议 |
| `\documentclass[journal,comsoc]{IEEEtran}` | Communications Society 期刊 |
| `\documentclass[9pt,technote]{IEEEtran}` | 技术简报 / Correspondence |

### 3.2 常用选项

| 选项 | 说明 |
|------|------|
| `lettersize` | 美式 Letter 纸张（8.5×11英寸） |
| `a4paper` | A4 纸张 |
| `draft` | 草稿模式（显示 overfull 警告标记） |
| `final` | 最终模式 |
| `review` | 审稿模式（加大行距，方便批注） |
| `10pt / 11pt / 12pt` | 字号（仅某些模式支持） |

**NMIGOD 论文推荐使用：**
```latex
\documentclass[lettersize,journal]{IEEEtran}
```

---

## 4. 前置信息（Front Matter）

### 4.1 标题

```latex
\title{论文标题}

% 如需手动换行：
\title{第一行标题 \\ 第二行标题}
```

### 4.2 作者与单位

**单作者：**
```latex
\author{Author Name,~\IEEEmembership{Member,~IEEE,}
\thanks{Author is with the Department of ... Email: xxx@xxx.edu.}}
```

**多作者（同一单位）：**
```latex
\author{Author One,~Author Two,~and~Author Three
\thanks{All authors are with ...}}
```

**多作者（不同单位）：**
```latex
\author{Author One\textsuperscript{1},~Author Two\textsuperscript{2}
\thanks{\textsuperscript{1}Department A, University X.}
\thanks{\textsuperscript{2}Department B, University Y.}}
```

### 4.3 页眉与出版信息

```latex
% 期刊页眉（左/右交替显示）
\markboth{Journal Name,~Vol.~XX, No.~XX, Month~2026}%
{Author Name: Paper Title}

% 出版 ID（期刊论文页面底部）
\IEEEpubid{0000--0000/00\$00.00~\copyright~2026 IEEE}
```

### 4.4 摘要与关键词

```latex
\begin{abstract}
This paper proposes a novel ...
\end{abstract}

\begin{IEEEkeywords}
Anomaly detection, graph convolutional network, mutual information.
\end{IEEEkeywords}
```

### 4.5 编译标题

```latex
\maketitle   % 必须在所有前置信息之后调用
```

---

## 5. 正文结构

### 5.1 章节标题

```latex
\section{一级标题}           % 如 I. Introduction
\subsection{二级标题}        % 如 A. Background
\subsubsection{三级标题}     % 如 1) Method Details

% 无编号标题（致谢、参考文献等）
\section*{Acknowledgments}
```

### 5.2 首字下沉（论文常见格式）

```latex
\IEEEPARstart{T}{his} is the first sentence of the introduction...
```
输出效果：首字母 "T" 下沉两行。

### 5.3 列表环境

**无序列表：**
```latex
\begin{itemize}
\item 第一项
\item 第二项
\end{itemize}
```

**有序列表：**
```latex
\begin{enumerate}
\item 第一步
\item 第二步
\end{enumerate}
```

**自定义列表：**
```latex
\begin{list}{}{}
\item 无标记项
\item 无标记项
\end{list}
```

### 5.4 双栏中的跨栏内容

```latex
% 跨双栏的宽图表（占据整页宽度）
\begin{figure*}[!t]
    ...
\end{figure*}

% 跨双栏的宽表格
\begin{table*}[!t]
    ...
\end{table*}
```

---

## 6. 图表插入

### 6.1 单栏图片

```latex
\begin{figure}[!t]                          % !t = 尽量放在页顶
\centering
\includegraphics[width=2.5in]{fig1}         % 宽度 2.5 英寸（单栏最大约 3.5in）
\caption{图片标题。}
\label{fig_example}                         % label 必须在 caption 之后
\end{figure}
```

### 6.2 双栏图片（跨栏，适合大图）

```latex
\begin{figure*}[!t]                         % figure* = 跨双栏
\centering
\includegraphics[width=\textwidth]{my_figure.pdf}
\caption{宽图片标题。}
\label{fig_wide}
\end{figure*}
```

### 6.3 子图（并排）

```latex
\begin{figure*}[!t]
\centering
\subfloat[子图A说明]{\includegraphics[width=2.5in]{fig_a}%
\label{fig_sub_a}}
\hfil
\subfloat[子图B说明]{\includegraphics[width=2.5in]{fig_b}%
\label{fig_sub_b}}
\caption{总标题：(a) 子图A, (b) 子图B。}
\label{fig_overall}
\end{figure*}
```

### 6.4 表格

**注意：IEEE 表格的 `\caption` 在表格上方！**

```latex
\begin{table}[!t]
\caption{表格标题（Title Case）\label{tab:example}}
\centering
\begin{tabular}{|c||c|}
\hline
列1 & 列2 \\
\hline
数据1 & 数据2 \\
\hline
\end{tabular}
\end{table}
```

**专业三线表（推荐）：**
```latex
\begin{table}[!t]
\caption{Comparison of Detection Performance\label{tab:performance}}
\centering
\begin{tabular}{lcccc}
\hline
Dataset & ADFNR & GCN & GCN-LOF & NMIGOD \\
\hline
abalone & 0.785 & 0.887 & 0.900 & 0.861 \\
adult   & 0.485 & 0.870 & 0.887 & 0.858 \\
\hline
Average & 0.708 & 0.855 & 0.863 & 0.882 \\
\hline
\end{tabular}
\end{table}
```

### 6.5 图片格式建议

| 格式 | 适用场景 |
|------|---------|
| **PDF** | 矢量图（最佳，无限缩放） |
| **EPS** | 矢量图（传统格式） |
| **PNG** | 位图/截图（≥300 DPI） |
| **JPG** | 照片 |

---

## 7. 数学公式

### 7.1 行内公式

```latex
The anomaly score $s_i$ is computed as $s_i = \sum_{j=1}^m w_j \cdot d_{ij}$.
```

### 7.2 单行公式（有编号）

```latex
\begin{equation}
\label{eq:main}
x'_a = \frac{x_a - \min(a)}{\max(a) - \min(a) + 10^{-8}}
\end{equation}
```
引用：`\ref{eq:main}` 或 `\eqref{eq:main}`（自动加括号）。

### 7.3 多行对齐公式

```latex
\begin{align}
a &= b + c \label{eq:align1} \\
d &= e + f + g + h \label{eq:align2}
\end{align}
```

### 7.4 分段函数（Cases）

```latex
\begin{equation}
z_m(t) = \begin{cases}
1, & \text{if } \beta_m(t) \geq \theta \\
0, & \text{otherwise.}
\end{cases}
\end{equation}
```

### 7.5 矩阵

```latex
% 方括号矩阵
\begin{equation}
\mathbf{M} = \begin{bmatrix}
m_{11} & m_{12} & \cdots & m_{1n} \\
m_{21} & m_{22} & \cdots & m_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
m_{n1} & m_{n2} & \cdots & m_{nn}
\end{bmatrix}
\end{equation}
```

### 7.6 求和、极限、上下标

```latex
\begin{equation}
\rho_a = 1 - \frac{NE_a}{\log_2|U|}, \quad
NE_a = -\sum_{i=1}^{|C|} \frac{|C_i|}{|U|} \log_2\frac{|C_i|}{|U|}
\end{equation}
```

### 7.7 数学字体速查

| 命令 | 输出 | 用途 |
|------|------|------|
| `\mathbf{x}` | **x** (粗体) | 向量/矩阵 |
| `\mathcal{L}` | ℒ (书法体) | 损失函数 |
| `\mathbb{R}` | ℝ (黑板体) | 实数集 |
| `\text{arg min}` | arg min (直立体) | 公式中的文字 |
| `\hat{y}` | ŷ | 估计量 |
| `\bar{x}` | x̄ | 均值 |

---

## 8. 算法伪代码

模板已加载 `algorithm` 和 `algorithmic` 宏包。

```latex
\begin{algorithm}[H]                       % H = 固定在当前位置
\caption{Your Algorithm Name}\label{alg:main}
\begin{algorithmic}
\STATE \textbf{Input:} Data $X$, parameter $\lambda$
\STATE \textbf{Output:} Anomaly scores $S$
\STATE
\STATE Initialize graph adjacency matrix $\mathbf{A}$
\FOR{$i = 1$ to $n$}
    \FOR{$j = 1$ to $n$}
        \STATE $\mathbf{A}_{ij} \gets \text{HEOM}(x_i, x_j)$
    \ENDFOR
\ENDFOR
\STATE $\mathbf{S} \gets \text{GCN}(\mathbf{A}, \mathbf{X})$
\STATE \textbf{return} $\mathbf{S}$
\end{algorithmic}
\end{algorithm}
```

**常用 algorithmic 命令：**

| 命令 | 含义 |
|------|------|
| `\STATE` | 普通语句 |
| `\IF{condition} ... \ENDIF` | 条件判断 |
| `\FOR{condition} ... \ENDFOR` | For 循环 |
| `\WHILE{condition} ... \ENDWHILE` | While 循环 |
| `\REQUIRE` | 输入要求 |
| `\ENSURE` | 输出保证 |
| `\RETURN` | 返回值 |

---

## 9. 参考文献

### 9.1 手动编写（推荐用于少量文献）

```latex
\begin{thebibliography}{99}       % 99 = 为编号预留宽度
\bibliographystyle{IEEEtran}

\bibitem{ref1}
K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image
recognition,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit.},
Las Vegas, NV, USA, 2016, pp. 770--778.

\bibitem{ref2}
T. N. Kipf and M. Welling, ``Semi-supervised classification with graph
convolutional networks,'' in \textit{Proc. ICLR}, Toulon, France, 2017.

\end{thebibliography}
```

### 9.2 BibTeX 方式（推荐用于大量文献）

1. 创建 `.bib` 文件（如 `references.bib`）：

```bibtex
@article{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={Proc. ICLR},
  year={2017}
}
```

2. 在 `.tex` 文件中引用：

```latex
\bibliographystyle{IEEEtran}
\bibliography{IEEEabrv,references}   % IEEEabrv 提供标准缩写
```

3. 编译流程：

```bash
pdflatex my_paper.tex
bibtex my_paper
pdflatex my_paper.tex
pdflatex my_paper.tex
```

### 9.3 IEEE 引用格式要点

| 文献类型 | 格式 |
|---------|------|
| 期刊论文 | `[#] Author(s), ``Title,'' \textit{Journal}, vol. X, no. X, pp. XX--XX, Year.` |
| 会议论文 | `[#] Author(s), ``Title,'' in \textit{Proc. Conf. Name}, City, Country, Year, pp. XX--XX.` |
| 书籍 | `[#] Author(s), \textit{Book Title}. City, Country: Publisher, Year.` |
| 在线资源 | `[#] Author(s), ``Title.'' [Online]. Available: URL` |

---

## 10. 作者传记

**带头像：**
```latex
\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,
    clip,keepaspectratio]{author_photo.jpg}}]{Author Name}
Biography text here. Include education, research interests, publications, etc.
\end{IEEEbiography}
```

**不带头像：**
```latex
\begin{IEEEbiographynophoto}{Author Name}
Biography text here.
\end{IEEEbiographynophoto}
```

---

## 11. 编译与排错

### 11.1 完整编译流程

```bash
# 基础编译（不包含文献）
pdflatex my_paper.tex
pdflatex my_paper.tex          # 第二次解决交叉引用

# 含 BibTeX 文献的完整编译
pdflatex my_paper.tex
bibtex my_paper
pdflatex my_paper.tex
pdflatex my_paper.tex
```

### 11.2 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `Undefined control sequence` | 宏包未加载 | 添加 `\usepackage{...}` |
| `File 'xxx.sty' not found` | 缺少宏包 | 通过 TeX Live Manager 安装 |
| `Label(s) may have changed` | 交叉引用未更新 | 再编译一次 |
| `Overfull \hbox` | 内容超出页面宽度 | 调整表格宽度或公式换行 |
| `! LaTeX Error: Unknown graphics extension: .svg` | 不支持 SVG | 转成 PDF 或 PNG |
| 中文乱码 | 未配置中文支持 | 用 XeLaTeX + `ctex` 或 `xeCJK` |

### 11.3 IEEE 投稿特殊注意事项

1. **不要修改页边距、行距、字体大小** — 模板已设定 IEEE 标准
2. **不要使用 `\vspace` 手动调整间距** — 会被编辑退回
3. **图片必须清晰** — 分辨率 ≥ 300 DPI
4. **参考文献格式严格** — 使用 `IEEEtran` bibliography style
5. **PDF 中嵌入所有字体** — 投稿系统会自动检查

---

## 12. 适配 NMIGOD 论文指南

### 12.1 论文结构建议

```latex
\section{Introduction}                                    % I. 引言
\section{Related Work}                                    % II. 相关工作
\section{Preliminaries}                                   % III. 预备知识
    \subsection{Neighborhood Mutual Information}
    \subsection{Graph Convolutional Networks}
\section{Proposed NMIGOD Method}                          % IV. 方法
    \subsection{Overall Framework}
    \subsection{Adaptive-Radius NMI Graph Construction}
    \subsection{GCN-Based Anomaly Detection}
\section{Experiments}                                     % V. 实验
    \subsection{Data Preprocessing}
    \subsection{Comparison Algorithms}
    \subsection{Evaluation Metrics}
    \subsection{Results and Analysis}
        \subsubsection{Overall Performance}
        \subsubsection{Ablation Study}
        \subsubsection{Statistical Significance}
\section{Conclusion}                                      % VI. 结论
\section*{Acknowledgments}                                % 致谢

\begin{thebibliography}{99}
    ...
\end{thebibliography}

% 作者传记
\begin{IEEEbiography}[{\includegraphics[width=1in...]{photo.jpg}}]{Author Name}
...
\end{IEEEbiography}
```

### 12.2 实验表格（数据来自 `metrics/` 目录）

```latex
\begin{table}[!t]
\caption{Precision at Best Threshold (per Dataset)\label{tab:precision}}
\centering
\footnotesize
\begin{tabular}{lcccccc}
\hline
Dataset & ADFNR & DASOD & GCN & GCN-LOF & NIEOD & NMIGOD \\
\hline
abalone & 0.2795 & 0.1500 & 0.7500 & 0.8824 & 0.2717 & \textbf{1.0000} \\
adult   & 0.0791 & 0.0889 & 0.3501 & 0.4259 & 0.0903 & 0.3068 \\
% ... 更多数据集 ...
\hline
Average & 0.3577 & 0.3266 & 0.5908 & 0.5846 & 0.3452 & \textbf{0.6283} \\
\hline
\end{tabular}
\end{table}
```

### 12.3 插入实验图像（SVG 需先转 PDF）

```bash
# 将 SVG 矢量图转为 PDF（推荐 inkscape）
inkscape combined_5x6_precision.svg --export-pdf=combined_5x6_precision.pdf

# 或将 PNG 直接用于草稿
# 直接用 images/ 目录下的 PNG 文件
```

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{combined_5x6_precision.pdf}
\caption{Precision curves for all 30 datasets (5×6 grid layout).}
\label{fig:precision_combined}
\end{figure*}
```

### 12.4 论文中的关键公式示例

```latex
% Min-Max 归一化 (公式 1)
\begin{equation}
\label{eq:minmax}
x'_a = \frac{x_a - \min(a)}{\max(a) - \min(a) + 10^{-8}}
\end{equation}

% 自适应半径 (公式 12 in paper)
\begin{equation}
\label{eq:adaptive_radius}
\varepsilon_a = \frac{\sigma_a}{1 + \rho_a}, \quad
\rho_a = 1 - \frac{NE_a}{\log_2|U|}
\end{equation}

% HEOM 距离
\begin{equation}
\label{eq:heom}
\text{HEOM}(x, y) = \sqrt{\sum_{a=1}^{m} d_a(x_a, y_a)^2}
\end{equation}

% NMI 图邻接矩阵
\begin{equation}
\label{eq:adjacency}
\mathbf{A}_{ij} = \begin{cases}
1, & \text{if } \text{HEOM}(x_i, x_j) \leq \varepsilon \text{ and } MI(x_i, x_j) > \tau \\
0, & \text{otherwise}
\end{cases}
\end{equation}
```

### 12.5 快速从 CSV 生成 LaTeX 表格

```python
import pandas as pd

# 读取 metrics CSV
df = pd.read_csv('metrics/precision.csv', index_col=0)

# 生成 LaTeX 表格
latex = df.to_latex(
    float_format="%.4f",
    bold_rows=True,
    column_format='l' + 'c' * len(df.columns),
    caption='Precision at Best Threshold',
    label='tab:precision',
)
print(latex)
```

---

## 附录：宏包速查表

| 宏包 | 命令 | 功能 |
|------|------|------|
| `amsmath` | `\begin{equation}` | 数学公式 |
| `amsfonts` | `\mathbb{R}` | 数学字体 |
| `graphicx` | `\includegraphics` | 插入图片 |
| `cite` | `\cite{ref1}` | 文献引用 |
| `subfig` | `\subfloat[]{}` | 子图并排 |
| `algorithm` | `\begin{algorithm}` | 算法浮动体 |
| `algorithmic` | `\STATE, \FOR` | 伪代码 |
| `url` | `\url{...}` | 超链接 |
| `hyperref` | `\hypersetup{...}` | PDF 书签（可选） |
| `booktabs` | `\toprule, \midrule` | 专业三线表（可选） |
