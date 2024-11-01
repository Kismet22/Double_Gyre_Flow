# MPC方法实现Double-gyre flow智能体路径规划



## 1 实现方法

### 1.1 实现工具

采用集成在Python语言环境中的**Casadi**工具包实现MPC控制方法；CasADi 是一个用于优化和自动微分的工具包，特别适用于动态系统的仿真和控制。

Casadi工具包的主要用途：

**符号计算：**支持符号表达式和矩阵的定义和操作，能够自动求导，适用于需要精确导数的应用场景。

**自动微分：**CasADi 提供了前向和反向模式的自动微分功能，这使得在优化过程中计算导数更加高效。

**优化求解：**支持各种类型的优化问题，包括非线性规划（NLP），二次规划（QP），以及带有约束的最优控制问题（OCP）。

**数值求解器：**内置了多种数值求解器，可以用于求解常微分方程（ODE），微分代数方程（DAE）等。



### 1.2 实现方法

**单射法（single-shooting）**是一种求解最优控制问题的数值方法。通过将整个时间区间离散化，并将最优控制问题转化为一个非线性规划（NLP）问题来求解。

实现过程

**1、初始条件定义**

初始状态：定义系统在初始时间点的状态 
$$
x(t0)=x0
$$
为已知的初始条件。

**2、控制变量定义**

控制输入：在整个时间区间内定义控制输入ut。在单射法中，控制输入通常在每个时间步内保持不变。

**3、时间离散化**

时间网格：将连续时间[t0,tf]进行离散。

**4、动态更新**

数值积分：使用数值积分方法在每个时间步上求解状态方程，对状态变量进行更新。

**5、优化目标**

目标函数：定义一个目标函数
![公式图1](https://github.com/user-attachments/assets/9192d0c7-958d-4d92-a509-0871208a4a60)

该目标函数由运行成本和终端成本两部分组成，运行成本在每步更新时累加，终端成本反映末状态和期望状态的关系。

**6、约束条件**

在问题构建过程中，状态变量和控制变量的值要在一定的范围内被约束。

**7、求解非线性规划问题**

**优化器**：将整个问题转化为一个 NLP 问题，求解最优的控制输入序列 u(t)，使用优化求解器（如 IPOPT）来求解这个 NLP 问题。



## 2 实现过程

### 2.1 动态更新模型建立

在Casadi语言环境下，建立智能体在Double-gyre flow中的状态更新模型，理论依据来自文献：

[1]Mei J, Kutz J N, Brunton S L. Observability-Based Energy Efficient Path Planning with Background Flow via Deep Reinforcement Learning[C]//2023 62nd IEEE Conference on Decision and Control (CDC). IEEE, 2023: 4364-4371.

[2]Gunnarson P, Mandralis I, Novati G, et al. Learning efficient navigation in vortical flow fields[J]. Nature communications, 2021, 12(1): 7143.



Double-gyre flow流场速度在文献[2]中定义：
$$
𝜓(𝑥,𝑦,𝑡) = 𝐴sin(𝜋 𝑓 (𝑥,𝑡))sin(𝜋𝑦)
$$

$$
𝑓 (𝑥,𝑡) = 𝜖 sin(𝜔𝑡)𝑥2 +𝑥−2𝜖sin(𝜔𝑡)𝑥
$$

$$
Uflow = v(𝑥,𝑦,𝑡)=[−𝜕𝜓/𝜕y,𝜕𝜓/𝜕x]=[−𝜋𝐴sin(𝜋𝑓(𝑥,𝑡))cos(𝜋𝑦), −𝜋𝐴cos(𝜋𝑓 (𝑥,𝑡))sin(𝜋𝑦)𝑑𝑓/𝑑𝑡]
$$

其中， 𝐴 = 0.5,𝜔 = 2𝜋,𝜖 = 0.25为论文中使用的参数。

智能体的运动公式在文献[1]中定义:
$$
X0=Xstart
$$

$$
Xn+1=Xn + Δt(Uswim[cos(θ),sin(θ)]+Uflow)
$$



根据上述公式，在Casadi的符号语言环境下编写代码段：

```
# 动力学函数
def f_1(pos_x, t):
    """

    param pos_x:          position x
    param t:        simulation time

    return: f(x,t)
    """
    out_put = epsilon * ca.sin(omega * t) * pos_x ** 2 + pos_x - 2 * epsilon * ca.sin(omega * t) * pos_x
    return out_put

def psi(pos_x, pos_y, t):
    """

    param pos_x:          position x
    param pos_y:          position y
    param t:        simulation time

    return: ψ(x,y,t)
    """

    out_put = A * ca.sin(ca.pi * f_1(pos_x, t)) * ca.sin(ca.pi * pos_y)
    return out_put

def U_flow_x(pos_x, pos_y, t):
    """

    param pos_x:          position x
    param pos_y:          position y
    param t:        simulation time

    return: vflow_x
    """

    out_put = -ca.pi * A * ca.sin(ca.pi * f_1(pos_x, t)) * ca.cos(ca.pi * pos_y)
    return out_put

def U_flow_y(pos_x, pos_y, t):
    """

    param pos_x:          position x
    param pos_y:          position y
    param t:        simulation time

    return: vflow_y
    """

    out_put = ca.pi * A * ca.cos(ca.pi * f_1(pos_x, t)) * ca.sin(ca.pi * pos_y) * (
            2 * epsilon * ca.sin(omega * t) * pos_x
            + 1 - 2 * epsilon * ca.sin(omega * t))
    return out_put
   
def agent_dynamics_withtime(state_init, control_init):
    action = control_init
    pos_x, pos_y, pos_t = state_init[0], state_init[1], state_init[2]
    v_flow_x = U_flow_x(pos_x, pos_y, pos_t)
    v_flow_y = U_flow_y(pos_x, pos_y, pos_t)

    delta_pos_x = U_swim * ca.cos(action) + v_flow_x
    delta_pos_y = U_swim * ca.sin(action) + v_flow_y
    delta_t = 1
    return ca.vertcat(delta_pos_x, delta_pos_y, delta_t)
```

在Casadi的语言环境下，

```
sin cos pi
```

需写为

```
ca.sin ca.cos ca.pi
```

'agent_dynamics_withtime' 定义了智能体的状态变化，智能体包含3个状态变量x、y、t以及一个控制变量θ。



### 2.2 Casadi环境下定义MPC问题和求解器

#### 2.2.1 环境变量声明

该部分主要声明在MPC方法过程中需要的参数，包括采样更新时间Δt、MPC方法需要预测的步数N以及开始的时间点t0。

```
dt = 0.1  # （模拟的）系统采样时间【秒】
N = 10  # 需要预测的步长【超参数】
t0 = 0  # 初始时间
```

其中，N是MPC过程中可以调节的变量，表示“之后N步的”控制输出



#### 2.2.2 Casadi控制问题建模

在该部分通过符号Casadi使用规范的方式，构建了一个包含状态、控制输入和更新方法的控制问题。

```
# 1 系统状态
x = ca.SX.sym('x')  # x坐标
y = ca.SX.sym('y')  # y坐标
t = ca.SX.sym('time')  # 时间步
states = ca.vertcat(x, y, t)
n_states = states.size()[0]  # 获得系统状态的尺寸，向量以（n_states, 1）的格式呈现

# 2 控制输入
theta = ca.SX.sym('theta')
controls = ca.vertcat(theta)
n_controls = controls.size()[0]  # 控制向量尺寸

# 3 运动学模型
# 定义右手函数
rhs = agent_dynamics_withtime(states, controls)
# 利用CasADi构建一个函数
f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
```

在Casadi的语言环境下，SX和MX是两种类型的符号矩阵：

**SX**：适用于稀疏符号矩阵（Sparse Matrix of Symbolic Expressions）。用于那些稀疏结构明确的符号计算，适合较小规模且结构固定的问题。具有较高的评估效率，但在符号操作和大规模计算时可能不如高效。

**MX**：适用于更复杂和更通用的符号矩阵（Matrix of Symbolic Expressions）允许稀疏和密集结构，并且在符号操作（如求导和简化）上更加灵活和高效。适用于大规模、结构复杂的优化和仿真问题。



```
f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
```

定义了一个符号函数映射关系，函数名称'f'，输入变量[states, controls]，映射到的右手函数在rhs方法中定义。



#### 2.2.3 MPC问题构建

该部分对MPC问题进行了构建，MPC问题预测某一时间步状态后数个时间步的运动状态更新过程。

**相关变量构建**

```
相关变量，格式为(状态长度， 步长)
U = ca.SX.sym('U', n_controls, N)  # N步内的控制输出
X = ca.SX.sym('X', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
P = ca.SX.sym('P', n_states + n_states)  # 构建问题的相关参数,在这里每次只需要给定当前/初始位置和目标终点位置
```

U和X包含N/N+1个动作信息/状态信息、P记录预测初始位置的信息和目标终点的信息。



**更新方式构建**

```
Single Shooting 约束条件
X[:, 0] = P[:n_states]
```

优化过程中，将状态变量的第一个值（初值）赋予P，P的后三个值为目标终点的值，不改变。

之后通过U中的控制变量对X中的内容进行更新：

```
#剩余N状态约束条件
for i in range(N):
    # 通过前述函数获得下个时刻系统状态变化。
    # 这里需要注意引用的index为[:, i]，因为X为(n_states, N+1)矩阵
    delta_X = f(X[:, i], U[:, i])  # delta_X
    X[:, i + 1] = X[:, i] + delta_X * dt
```

```
# 获得输入（控制输入，参数）和输出（系统状态）之间关系的函数ff
ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])
```

函数名'ff'，输入U和P，输出为X



**NLP问题的优化目标构建**

```
# NLP问题
# 位置惩罚矩阵
Q = np.zeros((2, 2))  # 全零 2x2 矩阵，用于惩罚位置
Q[:2, :2] = np.array([[10.0, 0.0],  # 对状态矩阵的前两个进行惩罚
                      [0.0, 10.0]])
# 时间惩罚矩阵
R = np.array([10.0])
# 优化目标
obj = 0  # 初始化优化目标值
for i in range(N):
    # 在N步内对获得优化目标表达式
    # .T表示矩阵转置,计算惩罚函数 对应误差的平方与系数相乘再相加
    # ca.mtimes,矩阵乘法操作
    obj = obj + ca.mtimes([(X[:2, i] - P[n_states:-1]).T, Q, X[:2, i] - P[n_states:-1]]) + \
          ca.mtimes([X[-1, i].T, R, X[-1, i]])
```

通过矩阵乘法的形式来构建位置优化目标，目标为与目标终点解决；通过惩罚时间状态的值来起到缩短运行时间的目的。



**状态变量的约束条件定义**

```
# 约束条件定义
g = []  # 用list来存储优化目标的向量
for i in range(N + 1):
    # 这里的约束条件只有小车的坐标（x,y）必须在-2至2之间
    # 由于xy没有特异性，所以在这个例子中顺序不重要（但是在更多实例中，这个很重要）
    # g中表示需要约束的内容
    # g.append(X[0, i])  # 第一行第n列
    # g.append(X[1, i])  # 第二行第n列
    pass
```

完成上述操作后，得到NLP问题的优化值函数

```
# 定义NLP问题，'f'为需要优化的目标函数，'x'为需要优化的目标变量，'p'为包含已知不变的参数，'g'为额外约束条件
# 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P, 'g': ca.vertcat(*g)}# .reshape(U, -1, 1):-1 表示该维度大小由另一维度推演得到, 1 一列
```

定义NLP问题，'f'为需要优化的目标函数，'x'为需要优化的目标变量，'p'为包含已知不变的参数，'g'为额外约束条件。分别对应obj、控制变量U、固定记录P和约束g



#### 2.2.4 求解器设置

采用'ipopt方法'作为MPC控制问题的求解器。**IPOPT（Interior Point OPTimizer）**使用的是内点法（Interior Point Method）来求解具有非线性目标函数和非线性约束的优化问题。内点法的核心思想是通过引入松弛变量，将不等式约束转化为等式约束，并且逐步缩小这些松弛变量，使解逐渐接近可行域的边界。

IPOPT方法的核心是将原始问题转化为一个无约束的问题，通过引入障碍函数：
$$
ϕ(x,μ)=f(x)−μ ∑ln(si)
$$
其中μ 是障碍参数，随着迭代进行逐渐减小。si 是松弛变量，满足 si>0。

IPOPT的方法的顺序如下：

1、**初始化**：选择初始解 x0、松弛变量 s0和障碍参数 μ0。

2、**KKT 条件**：求解 Karush-Kuhn-Tucker (KKT) 条件，这些条件包括了目标函数的梯度、等式约束和不等式约束的梯度，以及松弛变量和约束的互补条件。

3、**牛顿法**：使用修正的牛顿法来解决 KKT 系统，通过线性化的子问题来更新变量 x 和松弛变量 s。

4、**更新参数**：减小障碍参数 μ，并更新 x和 s。

5、**收敛性检查**：检查是否满足收敛标准（如梯度的大小、约束的违反程度等）。如果收敛，停止迭代；否则，返回步骤 2。



在代码段中，需要对IPOPT进行设置，并设置求解器

```
# ipopt设置:
# ipopt.max_iter: 最大迭代次数
# ipopt.print_level: 输出信息的详细级别，0 表示关闭输出
# print_time: 控制是否输出求解时间
# ipopt.acceptable_tol: 接受的KKT值的容忍度
# ipopt.acceptable_obj_change_tol: 接受的目标函数变化的容忍度
opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,'ipopt.acceptable_obj_change_tol': 1e-8}
```

其中'acceptable_tol'为可接受的最优性容忍度（acceptable tolerance），当 KKT（Karush-Kuhn-Tucker）条件的残差（误差）小于或等于这个值时，IPOPT 会认为当前解已经足够好，可以接受为一个可行解或次优解。这个参数用于设置 IPOPT 何时认为找到一个“足够好”的解，可以在某些情况下提前终止优化过程。设置一个较小的值意味着算法要求更高的精度，而较大的值则意味着算法可以在更低精度下终止。

'acceptable_obj_change_tol'为可接受的目标函数变化容忍度（acceptable objective change tolerance），当连续两次迭代之间的目标函数值变化小于或等于这个值时，IPOPT 会认为目标函数值已经足够稳定，可以接受当前解为一个可行解或次优解。



最终可以得到求解器：

```
# solver' 是求解器的名称
# ipopt' 指定了所使用的求解器为 IPOPT
# nlp_prob 是定义好的非线性优化问题
# opts_setting 是求解器的设置参数，告诉求解器如何进行求解
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
```



### 2.3 仿真过程

在开始仿真的过程中，先设置对控制条件的约束：

```
control_max = ca.pi
lbx = []  # 最低约束条件
ubx = []  # 最高约束条件
for _ in range(N):
    #   在一个循环里进行连续定义
    lbx.append(-control_max)
    ubx.append(control_max)

```

在对仿真所需的参数作进一步的设置：

```
# 仿真条件和相关变量
t0 = 0.0  # 仿真开始时间
x0 = np.array([1.5, 0.5, 0.0]).reshape(-1, 1)  # 初始始状态
xs = np.array([0.5, 0.5, np.nan]).reshape(-1, 1)  # 末状态，对末状态的时间条件不作要求
u0 = np.array([0.0] * N).reshape(-1, n_controls)  # 系统初始控制状态，为了统一本例中所有numpy有关,N行,n_controls列,每个值都是0，N为开始时设定的要预测的步数
# 变量都会定义成（N,状态数）的形式方便索引和print
x_c = []  # 存储系统的状态
position_record = []  # 记录坐标位置
u_c = []  # 存储控制全部计算后的控制指令
t_c = []  # 保存时间
xx = []  # 存储每一步位置
sim_time = 20  # 仿真时长
index_t = []  # 存储时间戳，以便计算每一步求解的时间
```



完成上述设定后，完整的仿真过程如下：

```
def shift_movement(delta_t, t_in, x_in, u, function):
    # 运动到下一个状态
    f_value = function(x_in, u[:, 0])
    st = x_in + delta_t * f_value
    # 时间增加
    t_move = t_in + delta_t
    # 准备下一个估计的最优控制
    # 更新控制输入矩阵，去掉第一列控制输入，在末尾添加上（复制）最后一列控制输入
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t_move, st, u_end.T
```



```
# 6 开始仿真
mpciter = 0  # 迭代计数器
start_time = time.time()  # 获取开始仿真时间
# 终止条件为目标的欧式距离小于D/6或者仿真超时
while np.linalg.norm(x0[:2] - xs[:2]) > D / 20 and mpciter - sim_time / dt < 0.0:
    print("'''''''''''''''''''''''''")
    print("mpc_iter", mpciter)
    print("控制器输入", u0)
    # 初始化优化参数
    # c_p中存储的是当前的位置信息和目标点的位置信息，对应函数设定中的P矩阵
    c_p = np.concatenate((x0, xs))
    # 初始化优化目标变量
    init_control = ca.reshape(u0, -1, 1)
    # 计算结果并且
    t_ = time.time()
    # 求解器得到控制输出
    res = solver(x0=init_control, p=c_p, lbx=lbx, ubx=ubx)
    index_t.append(time.time() - t_)
    # 获得最优控制结果u
    u_sol = ca.reshape(res['x'], n_controls, N)  # 记住将其恢复U的形状定义
    # 每一列代表了系统在每个时间步上的最优控制输入
    # 利用之前定义ff函数获得根据优化后的结果
    ff_value = ff(u_sol, c_p)  
    # 之后N+1步后的状态（n_states, N+1）
    # 存储结果
    x_c.append(ff_value)
    u_c.append(u_sol[:, 0])
    t_c.append(t0)
    # 根据数学模型和MPC计算的结果移动并且准备好下一个循环的初始化目标
    t0, x0, u0 = shift_movement(dt, t0, x0, u_sol, f)
    # 存储位置
    x0 = ca.reshape(x0, -1, 1)
    xx.append(x0.full())
    position_record.append([x0.full()[0], x0.full()[1]])
    # 打印状态值
    print("Current State:", x0.full())
    # 计数器+1
    mpciter = mpciter + 1
```



## 3 输出结果

调整预测的步数N，可以得到想要输出的结果：
![MPC-output](https://github.com/user-attachments/assets/9e9fa546-e542-4f19-b726-dd291524c83f)
