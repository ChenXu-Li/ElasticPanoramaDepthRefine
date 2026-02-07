你正在实现一个【单张全景图的深度矫正系统】。

目标：
给定一张全景图的初始深度 D0（来自 DAP）和一小撮稀疏但准确的参考点云，
在【log-depth 空间】中，求解一个各向异性的弹性位移场 Δ，
使得深度在参考点处被“钉死”，并且修正只在结构内部传播，
不会跨越楼宇边缘、天空边界或深度不连续处。

## 一、数学定义（必须严格遵守）

1. 所有优化变量在 log-depth 空间定义：

    z0(i) = log(D0(i))
    z(i)  = z0(i) + Δ(i)

   最终输出深度：
    D(i) = exp(z(i))

2. 优化变量：
    Δ ∈ R^N，N 为有效像素数

3. 图结构：
    - 节点：球面全景图的每个像素
    - 邻接：4-connected（经度方向周期 wrap）
    - 极区不做特殊处理（可 mask）

## 二、权重设计（关键）

1. 在 log-depth 空间计算梯度：

    ∇z0(i) = gradient(log(D0(i)))

2. 对相邻像素 (i, j)，定义各向异性弹性权重：

    w_ij = exp(
              - λ_g * ||∇z0(i) - ∇z0(j)||_2
           ) * exp(
              - λ_e * edge(i, j)
           )

其中：
- edge(i, j) = 1 表示跨越 sky / 语义边缘 / 深度断裂
- edge(i, j) = 0 表示同一连续表面

## 三、能量函数（严格二次、凸）

1. Anchor 项（log-depth）：

    L_anchor =
        Σ_k α_k * ( z0(i_k) + Δ(i_k) - log(D_ref_k) )^2

2. 各向异性弹性项（核心）：

    L_elastic =
        Σ_(i,j) w_ij * ( Δ(i) - Δ(j) )^2

3. 梯度保持项（log-depth）：

    L_grad =
        Σ_i || ∇(z0(i) + Δ(i)) - ∇z0(i) ||^2
      = Σ_i || ∇Δ(i) ||^2

4. 总能量：

    L = L_anchor
      + λ_elastic * L_elastic
      + λ_grad    * L_grad

## 四、数值求解

将能量写成线性系统：

    (L_elastic_matrix
     + λ_grad * L_grad_matrix
     + A_anchor) Δ = b_anchor

要求：
- 构建稀疏对称正定矩阵
- 使用 scipy.sparse + cg 或 spsolve
- 不使用深度学习框架

## 五、实现要求

- Python 3.9+
- 依赖：
    numpy
    scipy
    opencv-python
    imageio

- 输入：
    depth_dap.npy           # float32, H×W
    rgb.png                 # 可选，仅用于 edge / sky
    anchors.npy             # [K, 3] -> (theta, phi, depth)

- 输出：
    depth_refined.npy
    depth_refined.png（可选可视化）

## 六、你需要生成的文件

1. data/io.py
   - 读取 depth、anchor
   - 坐标与像素索引转换（θφ ↔ 像素）

2. geometry/sphere.py
   - 球面邻接
   - 梯度计算（log-depth）

3. graph/weights.py
   - 计算 w_ij
   - sky / edge mask 接口

4. solver/elastic_solver.py
   - 构建稀疏 Laplacian
   - 加入 anchor
   - 解线性系统

5. main.py
   - 串联完整流程
   - 保存结果

注意：
- 所有深度误差与梯度计算都在 log-depth 空间
- 不要实现 spline / grid / 神经网络
- 这是一个物理变形求解问题


建议目录
panorama_depth_refine/
│
├── data/
│   ├── io.py                # 数据读取 / 投影
│   └── example/
│       ├── depth_dap.npy
│       ├── anchors.npy
│       └── rgb.png
│
├── geometry/
│   ├── sphere.py            # 球面梯度、邻接
│   └── projection.py        # θφ ↔ pixel
│
├── graph/
│   ├── weights.py           # 各向异性 w_ij
│   └── laplacian.py         # 稀疏矩阵构建
│
├── solver/
│   └── elastic_solver.py    # 主要求解器
│
├── utils/
│   ├── visualization.py
│   └── masks.py             # sky / edge
│
├── main.py                  # 单图入口
├── config.yaml              # λ_g, λ_e, λ_grad 等
└── README.md
