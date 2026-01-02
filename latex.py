from typing import List, Mapping
from manim import *

class ME(Scene):
    """机器学习全部公式"""

    def Generate(self, map: Mapping[str,List[str]]):
        """迭代输出"""
        groups = VGroup()  # 每组是一个 VGroup(title, formulas)
        for title_str, formulas in map.items():
            # 2-1 标题
            title = Text(title_str, color=BLUE, font_size=28)
            # 2-2 公式列表
            tex_group = VGroup(*[
                MathTex(f, font_size=23)
                for f in formulas
            ]).arrange(DOWN, buff=0.5)
            # 2-3 标题在上、公式在下
            whole = VGroup(title, tex_group).arrange(DOWN, buff=0.7)
            groups.add(whole)

        # 4. 逐组播放：旧组淡出 → 新组淡入
        prev = None
        for g in groups:
            if prev is None:
                # 第一组直接写
                self.play(Write(g, run_time=0.5))
            else:
                self.play(
                    ReplacementTransform(prev, g, run_time=0.5)
                )
            prev = g

        # 5. 最后全部淡出
        self.play(FadeOut(prev))

    def LM(self):
        """线性模型"""
        map = {
            "多元线性模型": [
                r"\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b",
                r"\hat{y} = \mathbf{w}^T \mathbf{x} + b"
            ],
            "局部加权线性回归 LWR": [
                r"J(\theta) = \sum_{i=1}^{m} w^{(i)} \left( y^{(i)} - \theta^T x^{(i)} \right)^2",
                r"w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^2}{2k^2} \right)"
            ],
            "逻辑回归 Logistic Regression": [
                r"P(y=1 \mid x; \theta) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}",
                r"\log \left( \frac{P}{1-P} \right) = \theta^T x"
            ],
            "广义线性模型 GLM": [
                r"p(y \mid \eta) = b(y) \exp\left( \eta^T T(y) - a(\eta) \right)"
            ],
            "线性判别分析 LDA": [
                r"J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}"
            ],
            "高斯判别分析 GDA": [
                r"\ell(\phi, \mu_0, \mu_1, \Sigma) = \sum_{i=1}^{m} \log p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma)"
            ],
            "正规方程 Normal Equation": [
                r"\theta = (X^T X)^{-1} X^T y"
            ],
            "最小二乘法 OLS": [
                r"J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2",
                r"J(\theta) = \frac{1}{2} (X\theta - y)^T (X\theta - y)"
            ],
            "梯度下降 Gradient Descent": [
                r"\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)",
                r"\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}"
            ]
        }
        self.Generate(map)

    def DT(self):
        """决策树"""
        map = {
            "信息熵 Entropy": [
                r"H(D) = -\sum_{k=1}^{|\mathcal{Y}|} p_k \log_2 p_k"
            ],
            "条件熵 Conditional Entropy": [
                r"H(D|A) = \sum_{v=1}^{V} \frac{|D^v|}{|D|} H(D^v)"
            ],
            "信息增益 Information Gain": [
                r"Gain(D, A) = H(D) - H(D|A)"
            ],
            "基尼指数 Gini Index": [
                r"Gini(D) = 1 - \sum_{k=1}^{|\mathcal{Y}|} p_k^2",
                r"Gini\_index(D, A) = \sum_{v=1}^{V} \frac{|D^v|}{|D|} Gini(D^v)"
            ],
            "KL 散度 Relative Entropy": [
                r"D_{KL}(P\|Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}"
            ],
            "简化错误剪枝 REP": [
                r"E_{after} \le E_{before}"
            ],
            "悲观剪枝算法 PEP": [
                r"e'(t) = \frac{E(t) + \frac{1}{2}}{N(t)}",
                r"\text{If } E(t) \leq \sum_{l} E(l) + \frac{(\text{number of leaves in } T_t) - 1}{2}"
            ],
            "最小错误剪枝 MEP": [
                r"\mathrm{STE}(t)\;=\;\mathrm{EER}(t)\;=\;1-\hat p_{max}(t)",
                r"\mathrm{DYE}(t)\;=\;\sum_{c\in\mathrm{children}(t)} \frac{N(c)}{N(t)} \cdot \mathrm{EER}(c)",
                r"\mathrm{STE}(t)\;\le\;\mathrm{DYE}(t)"
            ],
            "代价复杂度剪枝 CCP": [
                r"R_\alpha(T) = R(T) + \alpha |\widetilde{T}|",
                r"\alpha_t = \frac{R(t) - R(T_t)}{|\widetilde{T_t}| - 1}"
            ]
        }
        self.Generate(map)

    def NN(self):
        """神经网络"""
        map = {
            "激活函数 Activations": [
                r"\sigma(x) = \frac{1}{1+e^{-x}}",
                r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
                r"f(x) = \max(0, x)",
                r"f(x) = \ln(1 + e^x)"
            ],
            "损失函数 Losses": [
                r"L = -\sum y_i \log(\hat{y}_i)",
                r"L_{focal} = -\alpha_t (1-\hat{p}_t)^\gamma \log(\hat{p}_t)"
            ],
            "优化与正则化 Optimization & Regularization": [
                r"\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t",
                r"y = \gamma \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + \beta",
                r"\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}",
                r"r_j \sim \text{Bernoulli}(p), \tilde{h} = r * h"
            ],
            "卷积神经网络 CNN": [
                r"O = \text{act}(\sum X * K + b)",
                r"(f * d g)(x) = \sum_{t} f(t)g(x - d \cdot t)",
                r"\text{Conv}_{dw} + \text{Conv}_{pw}",
                r"y_c = \frac{1}{H \times W} \sum_{i,j} x_{i,j,c}"
            ],
            "循环神经网络 RNN/LSTM/GRU": [
                r"h_t = \tanh(W_h h_{t-1} + W_x x_t + b)",
                r"f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)",
                r"C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_c[h_{t-1}, x_t] + b_c)",
                r"z_t = \sigma(W_z[h_{t-1}, x_t] + b_z)",
                r"h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t"
            ],
            "对抗神经网络与变分推断 GAN": [
                r"\min_G \max_D V(D, G) = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]"
            ],
            "Attention is all you need": [
                r"\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
                r"\text{head}_i = \text{Attn}(QW_i^Q, KW_i^K, VW_i^V)",
                r"PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})"
            ]
        }
        self.Generate(map)

    def SVM(self):
        """支持向量机 SVM"""
        map = {
            "支持向量机 Hard Margin": [
                r"\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2",
                r"\text{s.t. } y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad i=1, \dots, m",
                r"\text{Margin} = \frac{2}{\|\mathbf{w}\|}"
            ],
            "软间隔 Soft Margin SVM": [
                r"\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m \xi_i",
                r"\text{s.t. } y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0"
            ],
            "拉格朗日对偶 Lagrangian Duality": [
                r"L(\mathbf{w}, b, \alpha) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^m \alpha_i (y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1)",
                r"\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j",
                r"\alpha_i(y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1) = 0"
            ],
            "核函数与核技巧 Kernel Trick": [
                r"K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)",
                r"K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right)",
                r"f(\mathbf{x}) = \sum_{i=1}^m \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b"
            ],
            "正则化与 Hinge Loss": [
                r"\min_{f} \sum_{i=1}^m \max(0, 1 - y_i f(\mathbf{x}_i)) + \lambda \|\mathbf{w}\|^2",
                r"\ell_{0/1}(z) = \max(0, 1-z)"
            ],
            "支持向量回归 SVR": [
                r"\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^m (\xi_i + \xi_i^*)",
                r"\text{s.t. } |y_i - (\mathbf{w}^T \mathbf{x}_i + b)| \le \epsilon + \xi_i (\text{or } \xi_i^*)"
            ],
            "希尔伯特空间 Hilbert Space": [
                r"\langle f, g \rangle_{\mathcal{H}}",
                r"f(x) = \langle f, K(\cdot, x) \rangle_{\mathcal{H}}",
                r"\forall \{f_n\} \subset \mathcal{H} \text{ (Cauchy) } \implies f \in \mathcal{H}"
            ]
        }
        self.Generate(map)

    def Bayes(self):
        """贝叶斯分类器"""
        map = {
            "贝叶斯决策论 Bayesian Decision Theory": [
                r"P(c \mid \mathbf{x}) = \frac{P(c) P(\mathbf{x} \mid c)}{P(\mathbf{x})}",
                r"R(c_i \mid \mathbf{x}) = \sum_{j=1}^N \lambda_{ij} P(c_j \mid \mathbf{x})",
                r"h^*(\mathbf{x}) = \arg \min_{c \in \mathcal{Y}} R(c \mid \mathbf{x})"
            ],
            "极大似然估计 MLE": [
                r"L(\theta_c) = P(D_c \mid \theta_c) = \prod_{\mathbf{x} \in D_c} P(\mathbf{x} \mid \theta_c)",
                r"LL(\theta_c) = \sum_{\mathbf{x} \in D_c} \log P(\mathbf{x} \mid \theta_c)",
                r"\hat{\theta}_c = \arg \max_{\theta_c} LL(\theta_c)"
            ],
            "朴素贝叶斯 Naive Bayes": [
                r"P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^d P(x_i \mid c)",
                r"\hat{P}(c) = \frac{|D_c| + 1}{|D| + N} \quad (\text{Laplace Smoothing})",
                r"\hat{P}(x_i \mid c) = \frac{|D_{c, x_i}| + 1}{|D_c| + N_i}"
            ],
            "半朴素贝叶斯 Semi-Naive Bayes": [
                r"P(\mathbf{x} \mid c) = \prod_{i=1}^d P(x_i \mid c, pa_i) \quad (\text{ODE})",
                r"P(c \mid \mathbf{x}) \propto P(c) \sum_{i=1}^d P(x_i \mid c) \prod_{j=1}^d P(x_j \mid c, x_i) \quad (\text{AODE})"
            ],
            "贝叶斯网络 Bayesian Networks": [
                r"P(x_1, \dots, x_d) = \prod_{i=1}^d P(x_i \mid \pi_i)",
                r"f_{MSL}(\mathcal{G}, D) = \ln P(D \mid \hat{\theta}, \mathcal{G}) - \frac{d}{2} \ln m \quad (\text{BIC Score})"
            ],
            "EM 算法 Expectation-Maximization": [
                r"Q(\theta, \theta^t) = \mathbb{E}_{Z \mid X, \theta^t} [\ln P(X, Z \mid \theta)] \quad (\text{E-step})",
                r"\theta^{t+1} = \arg \max_{\theta} Q(\theta, \theta^t) \quad (\text{M-step})"
            ]
        }
        self.Generate(map)

    def Ensemble(self):
        """集成学习：结构化扩容版"""
        map = {
            "AdaBoost 权重与组合": [
                r"\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)",
                r"H(\mathbf{x}) = \text{sign}\left( \sum_{t=1}^T \alpha_t h_t(\mathbf{x}) \right)"
            ],
            "AdaBoost 分布更新": [
                r"D_{t+1}(i) = \frac{D_t(i)}{Z_t} \exp(-\alpha_t y_i h_t(\mathbf{x}_i))",
                r"Z_t = \sum_i D_t(i) \exp(-\alpha_t y_i h_t(\mathbf{x}_i))"
            ],
            "自助采样 Bootstrap": [
                r"\lim_{m \to \infty} (1 - \frac{1}{m})^m = \frac{1}{e} \approx 0.368",
                r"P(\text{sample was not selected}) \approx 36.8\%"
            ],
            "随机森林 Random Forest": [
                r"k = \log_2 d \quad (\text{or } \sqrt{d})",
                r"E_{oob} = \frac{1}{|D|} \sum_{(\mathbf{x},y) \in D} I(H_{oob}(\mathbf{x}) \neq y)"
            ],
            "平均法 mean": [
                r"H_{simple}(\mathbf{x}) = \frac{1}{T} \sum_{i=1}^T h_i(\mathbf{x})",
                r"H_{weight}(\mathbf{x}) = \sum_{i=1}^T w_i h_i(\mathbf{x})"
            ],
            "投票法 vote": [
                r"H_{vote}(\mathbf{x}) = c_{\arg \max_j \sum_{i=1}^T h_i^j(\mathbf{x})}",
                r"H_{weight\_vote}(\mathbf{x}) = c_{\arg \max_j \sum_{i=1}^T w_i h_i^j(\mathbf{x})}"
            ],
            "学习法 Stacking": [
                r"H(\mathbf{x}) = g(h_1(\mathbf{x}), h_2(\mathbf{x}), \dots, h_T(\mathbf{x}))"
            ],
            "误差—分歧分解 E-A Decomposition": [
                r"\bar{A} = \sum_{i=1}^T w_i \int (h_i(x) - H(x))^2 p(x) dx",
                r"E = \bar{E} - \bar{A}"
            ],
            "Q-统计量与不合度量": [
                r"Q_{ij} = \frac{ad-bc}{ad+bc}",
                r"dis_{ij} = \frac{b+c}{a+b+c+d}"
            ],
            "相关系数": [
                r"\rho_{ij} = \frac{ad-bc}{\sqrt{(a+b)(a+c)(c+d)(b+d)}}"
            ]
        }
        self.Generate(map)

    def Clustering(self):
        """聚类算法：结构化精简版"""
        map = {
            "Jaccard 与 FM 指数": [
                r"JC = \frac{a}{a+b+c}",
                r"FMI = \sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}"
            ],
            "Rand 指数 RI": [
                r"RI = \frac{a+d}{a+b+c+d}",
                r"\text{Note: } a,b,c,d \text{Statistics for pairs of samples of the same/different classes}"
            ],
            "DB 指数 DBI": [
                r"DBI = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left( \frac{avg(C_i) + avg(C_j)}{d_{cen}(\mu_i, \mu_j)} \right)",
                r"avg(C) = \frac{2}{|C|(|C|-1)} \sum_{1 \le i < j \le |C|} dist(x_i, x_j)"
            ],
            "Dunn 指数 DI": [
                r"DI = \min_{1 \le i \le k} \left\{ \min_{j \neq i} \left( \frac{d_{min}(C_i, C_j)}{\max_{1 \le l \le k} diam(C_l)} \right) \right\}",
                r"diam(C) = \max_{x, y \in C} dist(x, y)"
            ],
            "闵可夫斯基/欧式/曼哈顿": [
                r"dist_{mk}(x_i, x_j) = \left( \sum_{u=1}^n |x_{iu} - x_{ju}|^p \right)^{1/p}",
                r"\text{Euclidean: } p=2; \quad \text{Manhattan: } p=1"
            ],
            "马氏/VDM 距离": [
                r"dist_{mah}(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{(\mathbf{x}_i - \mathbf{x}_j)^T \mathbf{\Sigma}^{-1} (\mathbf{x}_i - \mathbf{x}_j)}",
                r"VDM_p(a, b) = \sum_{i=1}^k | \frac{m_{u,a,i}}{m_{u,a}} - \frac{m_{u,b,i}}{m_{u,b}} |^p"
            ],
            "k-Means 目标函数": [
                r"E = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2",
                r"\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x"
            ],
            "学习向量量化 LVQ": [
                r"p' = p + \eta(x - p) \quad (\text{if label match})",
                r"p' = p - \eta(x - p) \quad (\text{if label mismatch})"
            ],
            "高斯混合聚类 GMM": [
                r"P(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu))",
                r"\gamma_{ji} = \frac{\alpha_i \cdot p(x_j \mid \mu_i, \Sigma_i)}{\sum_{l=1}^k \alpha_l \cdot p(x_j \mid \mu_l, \Sigma_l)}"
            ],
            "密度聚类 DBSCAN": [
                r"N_\epsilon(x) = \{y \in D \mid dist(x, y) \le \epsilon\}",
                r"x \text{ is Core Point if } |N_\epsilon(x)| \ge \text{MinPts}"
            ],
            "层次聚类 AGNES": [
                r"d_{min}(C_i, C_j) = \min_{x \in C_i, y \in C_j} dist(x, y)",
                r"d_{avg}(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} dist(x, y)"
            ]
        }
        self.Generate(map)

    def DR(self):
        """降维与度量学习"""
        map = {
            "k-近邻 k-Nearest Neighbors": [
                r"h(\mathbf{x}) = \text{mode} \{ y_i \mid \mathbf{x}_i \in N_k(\mathbf{x}) \}",
                r"\text{Bayes Error Rate: } P^* \le P \le 2P^* (1 - P^*)"
            ],
            "多维尺度变换 MDS": [
                r"b_{ij} = -\frac{1}{2}(d_{ij}^2 - dist_{i\cdot}^2 - dist_{\cdot j}^2 + dist_{\cdot\cdot}^2)",
                r"B = \mathbf{Z}^T \mathbf{Z} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T"
            ],
            "主成分分析 PCA": [
                r"\max_{\mathbf{W}} \text{tr}(\mathbf{W}^T \mathbf{X}\mathbf{X}^T \mathbf{W}) \quad \text{s.t. } \mathbf{W}^T\mathbf{W} = \mathbf{I}",
                r"\mathbf{\Sigma}\mathbf{w}_i = \lambda_i \mathbf{w}_i \quad (\mathbf{\Sigma} = \mathbf{X}\mathbf{X}^T)"
            ],
            "核主成分分析 KPCA": [
                r"\mathbf{K}\mathbf{a}_i = \lambda_i \mathbf{a}_i",
                r"z_i = \sum_{j=1}^m \alpha_j^i K(\mathbf{x}_j, \mathbf{x})"
            ],
            "等度量映射 Isomap": [
                r"dist_{geo}(x_i, x_j) = \min (\text{Dijkstra / Floyd Paths})",
                r"\text{Input: } \mathbf{D}_{geo} \to \text{Output: } MDS(\mathbf{D}_{geo})"
            ],
            "权重重构 LLE": [
                r"\min_{w_1, \dots, w_k} \| \mathbf{x}_i - \sum_{j \in N(i)} w_{ij} \mathbf{x}_j \|^2",
                r"\text{s.t. } \sum_{j \in N(i)} w_{ij} = 1"
            ],
            "低维映射 LLE": [
                r"\min_{\mathbf{y}_1, \dots, \mathbf{y}_m} \sum_{i=1}^m \| \mathbf{y}_i - \sum_{j \in N(i)} w_{ij} \mathbf{y}_j \|^2",
                r"\mathbf{M} = (\mathbf{I}-\mathbf{W})^T(\mathbf{I}-\mathbf{W})"
            ],
            "度量学习 Metric Learning": [
                r"d_{\mathbf{M}}(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T \mathbf{M} (\mathbf{x}-\mathbf{y})}",
                r"\mathbf{M} = \mathbf{L}^T \mathbf{L} \quad (\mathbf{M} \text{ is PSD})"
            ]
        }
        self.Generate(map)

    def construct(self):
        part = Text("线性模型")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.LM()

        part = Text("决策树")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.DT()

        part = Text("神经网络")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.NN()

        part = Text("支持向量机")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.SVM()

        part = Text("贝叶斯分类器")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.Bayes()

        part = Text("集成学习")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.Ensemble()

        part = Text("聚类算法")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.Clustering()

        part = Text("降维与度量学习")
        self.play(Write(part, run_time=0.5))
        self.play(Unwrite(part, run_time=0.5))
        self.DR()