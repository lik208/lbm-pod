// lbm_pod_cpu.cpp
// ====================================================================
// 简化版 纯CPU POD（半拉格朗日）格子玻尔兹曼方法求解器
// 用于 D3Q27 格子上的剪切波衰减测试
//
// 原始代码: https://github.com/zakirovandrey/lbm-pod (CUDA GPU版本)
//
// 编译:
//   g++ -O2 -std=c++14 -o lbm_pod_cpu lbm_pod_cpu.cpp -lm
//
// 运行:
//   ./lbm_pod_cpu [Nx] [Ny] [Nz] [MaxSteps] [RegOrder]
//
// 默认: Nx=Ny=Nz=20, MaxSteps=100, RegOrder=2
// RegOrder: -1 = 全矩阵求逆方法, >=0 = Hermite张量正则化方法
//
// ====================================================================
// 算法概述 (POD = Particles on Demand, 半拉格朗日LBM):
//
// 传统LBM中，离散速度方向是整数格点坐标（如D3Q27中的0,±1），
// 粒子恰好从一个格点传播到相邻格点，不需要插值。
//
// POD方法中，离散速度方向是浮点数（如±sqrt(3)），粒子的出发点
// 不再落在格点上，因此需要插值重构。这提供了更高的精度和灵活性，
// 但也增加了计算复杂度。
//
// 每个时间步的流程:
//   1. 对每个格点(ix,iy,iz)，进行内迭代（定点迭代）直到收敛：
//      a. 从当前宏观量(ρ, vel, T)构建"gauge"（标度参数）
//      b. 对每个离散速度方向iq:
//         - 计算实际速度 v_iq = ef[iq]*sqrt(T/TLat) + vel
//         - 沿v_iq回溯到出发点（半拉格朗日特征线追踪）
//         - 在出发点周围用Lagrange多项式插值重构分布函数
//           (在矩空间或Hermite张量空间进行插值)
//      c. 从新的f计算宏观量 ρ, vel, T
//      d. 检查收敛性（新旧宏观量之差是否小于阈值）
//   2. 碰撞: BGK松弛 f → f - (1/τ)(f - f^eq)
//      POD中平衡态用零速度和格子温度TLat计算，
//      因为平流已在半拉格朗日步中完成
//   3. 将结果写入另一个缓冲区（双缓冲交替读写）
// ====================================================================

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <chrono>

// ============================================================
//  三维向量类型 (替代CUDA的float3/double3)
// ============================================================
struct Vec3 {
    double x, y, z;
};

// 向量运算符重载
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator*(double s, Vec3 a) { return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator/(Vec3 a, double s) { return {a.x/s, a.y/s, a.z/s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

// ============================================================
//  D3Q27 POD 格子常数
//
//  POD方法的核心特征：离散速度不是整数而是浮点数。
//  D3Q27_POD中，每个速度分量为 0 或 ±sqrt(3)。
//
//  格子温度 TLat = 1（而非传统LBM的1/3），
//  声速平方 cs2 = TLat = 1。
//
//  27个速度方向按壳层分为4类:
//    - 静止方向 (1个):  速度=0, 权重=8/27
//    - 面方向   (6个):  |v|=sqrt(3), 权重=2/27
//    - 棱方向   (12个): |v|=sqrt(6), 权重=1/54
//    - 角方向   (8个):  |v|=3, 权重=1/216
// ============================================================
static const int DIM = 3;     // 空间维度
static const int QN  = 27;    // 离散速度方向数

static const double TLat = 1.0;         // 格子温度 (POD中为1, 不是传统的1/3)
static const double cs2  = TLat;        // 声速平方 = 格子温度
static const double dcs2 = 1.0 / cs2;   // 声速平方的倒数

// 离散速度分量大小 = sqrt(3) ≈ 1.732
// 这是POD方法的关键: 速度是无理数, 不落在整数格点上
static const double EC = std::sqrt(3.0);

static const double M_PI = 4.E0 * atan(1.E0);

// 27个离散速度方向 ef[i] (浮点坐标)
// 顺序: 静止(1), 面(6), 棱(12), 角(8)
static const Vec3 ef[QN] = {
    // 静止方向 (v=0)
    { 0,   0,   0  },
    // 面方向: 沿坐标轴, 速度分量为 ±sqrt(3)
    {+EC,  0,   0  }, {-EC,  0,   0  },   // ±x
    { 0,  +EC,  0  }, { 0,  -EC,  0  },   // ±y
    { 0,   0,  +EC }, { 0,   0,  -EC },   // ±z
    // 棱方向: 沿两个坐标轴对角线, 两个非零分量各为 ±sqrt(3)
    {+EC, +EC,  0  }, {+EC,  0,  +EC }, { 0,  +EC, +EC },   // ++xy, ++xz, ++yz
    {+EC, -EC,  0  }, {+EC,  0,  -EC }, { 0,  +EC, -EC },   // +-xy, +-xz, +-yz
    {-EC, +EC,  0  }, {-EC,  0,  +EC }, { 0,  -EC, +EC },   // -+xy, -+xz, -+yz
    {-EC, -EC,  0  }, {-EC,  0,  -EC }, { 0,  -EC, -EC },   // --xy, --xz, --yz
    // 角方向: 沿体对角线, 三个分量各为 ±sqrt(3)
    {+EC, +EC, +EC }, {-EC, +EC, +EC }, {+EC, -EC, +EC }, {+EC, +EC, -EC },
    {-EC, -EC, +EC }, {-EC, +EC, -EC }, {+EC, -EC, -EC }, {-EC, -EC, -EC }
};

// 格子权重: 按壳层分配, 满足离散速度矩的精度要求
static const double W0 = 8.0 / 27.0;    // 静止方向权重 ≈ 0.2963
static const double W1 = 2.0 / 27.0;    // 面方向权重   ≈ 0.0741
static const double W2 = 1.0 / 54.0;    // 棱方向权重   ≈ 0.0185
static const double W3 = 1.0 / 216.0;   // 角方向权重   ≈ 0.0046

static const double wt[QN] = {
    W0,                                                      // 1个静止
    W1, W1, W1, W1, W1, W1,                                 // 6个面
    W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2,        // 12个棱
    W3, W3, W3, W3, W3, W3, W3, W3                          // 8个角
};

// 矩阵方法(RegOrder<0)所需的矩幂次数组
// 第i个矩定义为: mom_i = Σ_j v_j^{px} * v_j^{py} * v_j^{pz} * f_j
// 其中 (px, py, pz) = MomentsPower[i]
// 27个矩覆盖 {0,1,2}^3 的全部组合，恰好与27个方向匹配
static const int MomentsPower[QN][3] = {
    // px,py,pz — 按 z慢/y中/x快 的顺序排列
    {0,0,0},{1,0,0},{2,0,0}, {0,1,0},{1,1,0},{2,1,0}, {0,2,0},{1,2,0},{2,2,0},  // pz=0
    {0,0,1},{1,0,1},{2,0,1}, {0,1,1},{1,1,1},{2,1,1}, {0,2,1},{1,2,1},{2,2,1},  // pz=1
    {0,0,2},{1,0,2},{2,0,2}, {0,1,2},{1,1,2},{2,1,2}, {0,2,2},{1,2,2},{2,2,2},  // pz=2
};

// ============================================================
//  辅助函数: 整数幂 (避免调用通用pow)
// ============================================================
inline double ipow(double s, int n) {
    switch(n) {
        case 0: return 1.0;
        case 1: return s;
        case 2: return s * s;
        case 3: return s * s * s;
        default: return std::pow(s, n);
    }
}

// ============================================================
//  格点单元结构体
//  存储一个格点的全部信息:
//    f[27]  — 27个方向的分布函数值
//    rho    — 密度 (f的零阶矩: ρ = Σ f_i)
//    vel    — 宏观速度 (f的一阶矩除以密度)
//    T      — 温度 (由f的二阶矩推导)
// ============================================================
struct Cell {
    double f[QN];   // 分布函数 (27个方向)
    double rho;     // 宏观密度
    Vec3   vel;     // 宏观速度
    double T;       // 温度
};

// ============================================================
//  物理参数结构体
// ============================================================
struct PhysParams {
    // --- LBM核心参数 ---
    double tau;     // 松弛时间 τ (控制粘度: ν = cs² * (τ - 0.5) * dt)
    double dtau;    // 1/τ (碰撞频率, 用于BGK碰撞算子)
    double visc_atT;// 输入粘度值 (格子单位)

    // --- 格子单位 ---
    double dr;      // 空间步长 (默认=1)
    double dt;      // 时间步长 (默认=1)

    // --- 插值参数 ---
    int stencilInterpWidth;  // Lagrange插值模板宽度 (=2时使用3点插值)
    int stencilFixed;        // =1: 固定模板中心在当前格点; =0: 自适应居中于出发点

    // --- 正则化参数 ---
    int RegOrder;    // 正则化阶数:
                     //   -1: 全矩阵求逆法 (保留所有矩, 精度高但计算量大)
                     //   >=0: Hermite张量正则化法 (只保留到第N阶矩, 截断高阶)
                     //   常用值: 2 (保留密度+动量+应力张量)

    // --- 平衡态参数 ---
    int EquilibriumOrder;      // 平衡态Hermite展开阶数 (1~4阶)
    int IsothermalRelaxation;  // =1: 等温松弛(Tcur=cs²); =0: 非等温(Tcur=T)

    // --- 模拟控制 ---
    int fixedTemperature;   // =1: 固定温度不随时间演化
    int MaxSteps;           // 最大时间步数
    int StepIterPeriod;     // 每隔多少步输出诊断信息

    // --- 剪切波初始条件参数 ---
    double u0;      // 剪切波速度振幅
    double rho0;    // 初始密度
    double T0;      // 初始温度
    double uDragX, uDragY, uDragZ;  // 背景拖拽速度 (均匀流速)
    int    shearWaveDir;    // 剪切波方向: 1=X, 2=XY对角, 3=XYZ对角

    // 根据粘度计算松弛时间
    // τ = 0.5 + ν / (cs² * dt) = 0.5 + visc_atT * (1/dt)
    void setupUnits() {
        double ViscAtTUnitConv = 1.0 / dt;
        tau  = 0.5 + visc_atT * ViscAtTUnitConv;
        dtau = 1.0 / tau;
    }
};

// ============================================================
//  全局状态变量
// ============================================================
static int Nx, Ny, Nz;                  // 网格尺寸
static std::vector<Cell> cells[2];      // 双缓冲: cells[0]=旧时刻, cells[1]=新时刻
static PhysParams PP;                   // 物理参数

// 三维索引 → 一维数组索引 (行优先, x最快变化)
inline int idx(int ix, int iy, int iz) {
    return ix + iy * Nx + iz * Nx * Ny;
}

// 周期性边界条件: 将索引i映射到[0, N)
inline int wrap(int i, int N) { return ((i % N) + N) % N; }

// ============================================================
//  平衡态分布函数 f^eq (Maxwell-Boltzmann分布的Hermite展开)
//
//  feq_i = w_i * ρ * Σ_{n=0}^{eno} (1/n!) * H_n(e_i·u, T)
//
//  这里使用直接展开到4阶的形式:
//    0阶: 1
//    1阶: + (e_i·u) / cs²
//    2阶: + (e_i·u)² / (2cs⁴) - u² / (2cs²) + 温度修正项
//    3阶: + 1/6 * (e_i·u) * [高阶项]
//    4阶: + 1/24 * [高阶项]
//
//  参数:
//    Rho      — 密度
//    u        — 宏观速度
//    Tempr    — 温度 (等温模式下不使用)
//
//  对应原始代码: data-inl.cu :: Cell::calcEq()
// ============================================================
static void calcEq(double feq[QN], double Rho, Vec3 u, double Tempr) {
    if (Rho == 0) u = {0, 0, 0};  // 零密度时速度无意义

    const double dT  = dcs2;           // 1/cs² (对POD: =1)
    const double T0  = cs2;            // 格子参考温度 (=TLat=1)
    const double dT2 = dT * dT;       // (1/cs²)²
    const double dT4 = dT2 * dT2;     // (1/cs²)⁴
    const double u2  = dot(u, u);      // |u|²

    const int eno = PP.EquilibriumOrder;  // 展开阶数
    // 各阶开关 (阶数不够则对应项为0)
    const int T1 = (eno >= 1), T2 = (eno >= 2), T3 = (eno >= 3), T4 = (eno >= 4);

    // 等温松弛: Tcur = cs² (忽略实际温度波动)
    // 非等温:   Tcur = Tempr (使用实际温度)
    double Tcur = PP.IsothermalRelaxation ? cs2 : Tempr;

    // 与速度无关的基准项 (所有方向共享)
    const double mxwU = 1.0 - T2 * u2 * 0.5 * dT;

    for (int i = 0; i < QN; i++) {
        const Vec3 ei = ef[i];             // 第i个离散速度方向
        const double ei2 = dot(ei, ei);    // |e_i|²
        const double ei4 = ei2 * ei2;      // |e_i|⁴
        const double eu  = dot(ei, u);     // e_i · u (速度在e_i方向的投影)
        const double eu2 = eu * eu;        // (e_i · u)²
        const double eu4 = eu2 * eu2;      // (e_i · u)⁴
        const double dTc = Tcur - T0;      // 温度偏差 (等温时=0)

        // Hermite展开的各阶贡献
        double mxw = mxwU
            // 1阶: 线性速度项 (产生动量)
            + T1 * eu * dT
            // 2阶: 二次速度项 (产生应力张量) + 温度修正
            + T2 * eu2 * 0.5 * dT2
            + T2 * dTc * 0.5 * dT * (ei2 * dT - DIM)
            // 3阶: 三次速度项 (产生热流)
            + T3 * (1.0/6.0) * eu * dT * (eu2*dT2 - 3*u2*dT + 3*dTc*dT*(ei2*dT - DIM - 2))
            // 4阶: 四次速度项 (更高阶非平衡效应)
            + T4 * (1.0/24.0) * dT4 * (
                eu4 + 3*T0*T0*u2*u2 - 6*T0*eu2*u2
                + 6*dTc*eu2*ei2 + 3*dTc*dTc*ei4
                - 6*T0*dTc*dTc*(DIM+2)*ei2 + 3*T0*T0*dTc*dTc*DIM*(DIM+2)
                - 6*T0*dTc*u2*ei2 - 6*T0*dTc*(DIM+4)*eu2 + 6*T0*T0*dTc*(DIM+2)*u2
              );

        // 最终: feq_i = w_i * ρ * (Hermite展开系数)
        feq[i] = wt[i] * Rho * mxw;
    }
}

// ============================================================
//  BGK碰撞算子
//
//  f_new = f - (1/τ) * (f - f^eq)
//
//  这是LBM中最简单的碰撞模型 (Bhatnagar-Gross-Krook)。
//  物理含义: 分布函数以时间尺度τ向平衡态松弛。
//  τ越大 → 粘度越大 → 松弛越慢。
//
//  注意: 在POD中, f^eq使用零速度和格子温度TLat计算,
//  因为半拉格朗日步已经处理了平流(对流)。
// ============================================================
static void collision(double f[QN], const double feq[QN]) {
    const double dtau = PP.dtau;  // 1/τ
    for (int i = 0; i < QN; i++)
        f[i] = f[i] - dtau * (f[i] - feq[i]);
}

// ============================================================
//  收敛性检查 (用于内迭代)
//
//  POD方法中, gauge(标度参数)依赖于当前格点的宏观量,
//  而宏观量又取决于插值结果——这形成了一个非线性耦合。
//  因此需要迭代求解, 直到前后两次迭代的结果足够接近。
//
//  判据: |new - old| < err_abs + err_rel * |old|
//  对所有宏观量(ρ, vel, T)和分布函数f都要满足。
//
//  对应原始代码: data.cuh :: isConv()
// ============================================================
static bool isConv(const Cell& c1, const Cell& c2) {
    const double err_abs = 1e-12;   // 绝对误差阈值
    const double err_rel = 1e-10;   // 相对误差阈值
    // 检查宏观量: ρ, vx, vy, vz, T
    const double v1[] = {c1.rho, c1.vel.x, c1.vel.y, c1.vel.z, c1.T};
    const double v2[] = {c2.rho, c2.vel.x, c2.vel.y, c2.vel.z, c2.T};
    for (int i = 0; i < 5; i++)
        if (std::fabs(v1[i]-v2[i]) >= err_abs + err_rel*std::fabs(v1[i])) return false;
    // 检查所有27个方向的分布函数
    for (int i = 0; i < QN; i++)
        if (std::fabs(c1.f[i]-c2.f[i]) >= err_abs + err_rel*std::fabs(c1.f[i])) return false;
    return true;
}

// ============================================================
//  Lagrange插值多项式 (三维张量积形式)
//
//  用于半拉格朗日回溯: 出发点通常不落在格点上,
//  需要从周围格点的值插值得到出发点处的值。
//
//  三维Lagrange插值 = L_x(ix) * L_y(iy) * L_z(iz)
//  其中每个一维Lagrange基函数:
//    L_x(ix) = Π_{p≠ix} (shifts.x - p) / (ix - p)
//
//  参数:
//    ix,iy,iz — 模板内的局部索引 (0 到 N-1)
//    shifts   — 出发点在模板内的分数坐标
//    N        — 每个方向的插值点数 (= stencilInterpWidth + 1)
//
//  对应原始代码: streaming-pod.cu :: LagrPol()
// ============================================================
static double LagrPol(int ix, int iy, int iz, Vec3 shifts, int N) {
    double a = 1.0;
    // x方向Lagrange基函数
    for (int p = 0; p < N; p++) if (p != ix) a *= (shifts.x - p) / (ix - p);
    // y方向Lagrange基函数
    for (int p = 0; p < N; p++) if (p != iy) a *= (shifts.y - p) / (iy - p);
    // z方向Lagrange基函数
    for (int p = 0; p < N; p++) if (p != iz) a *= (shifts.z - p) / (iz - p);
    return a;
}

// ============================================================
//  Hermite多项式 H_n(v; a,b,c,d,e)
//
//  概率论者Hermite张量多项式, 用于将分布函数展开为矩。
//  这些多项式关于标准正态分布正交。
//
//  定义 (v是速度向量, a,b,c,...是分量索引 ∈{0,1,2}={x,y,z}):
//    H_0 = 1                                              (标量)
//    H_1(a) = v_a                                         (向量)
//    H_2(a,b) = v_a*v_b - δ_ab                            (二阶张量)
//    H_3(a,b,c) = v_a*v_b*v_c - 全部δ项                   (三阶张量)
//    H_4(a,b,c,d) = v_a*v_b*v_c*v_d - 全部δ项 + 全部δδ项  (四阶张量)
//
//  其中 δ_ab 是Kronecker delta (a==b时=1, 否则=0)。
//
//  对应原始代码: momentsMatrix.cuh :: Hermite()
// ============================================================
static double Hermite(int ni, const double v[3], const int abc[5]) {
    const int a = abc[0], b = abc[1], c = abc[2], d = abc[3];
    if (ni == 0) return 1.0;
    if (ni == 1) return v[a];
    if (ni == 2) return v[a]*v[b] - (a == b);
    if (ni == 3) return v[a]*v[b]*v[c]
        - (v[a]*(b==c) + v[b]*(a==c) + v[c]*(a==b));
    if (ni == 4) return v[a]*v[b]*v[c]*v[d]
        - (v[a]*v[b]*(c==d) + v[a]*v[c]*(b==d) + v[a]*v[d]*(b==c)
         + v[b]*v[c]*(a==d) + v[b]*v[d]*(a==c) + v[c]*v[d]*(a==b))
        + ((a==b)*(c==d) + (a==c)*(b==d) + (a==d)*(b==c));
    assert(false);
    return 0;
}

// ============================================================
//  Hermite张量系数存储结构
//
//  存储Hermite展开的各阶系数 a_n^{abc...}:
//    0阶: 1个标量   (密度)
//    1阶: 3个分量   (动量)
//    2阶: 9个分量   (应力张量)
//    3阶: 27个分量  (热流张量)
//    ...
//    总数 = Σ_{k=0}^{Order} 3^k
//
//  例如 Order=2: 1+3+9 = 13个系数
//
//  对应原始代码: momentsMatrix.cuh :: TensorCoeffs<>
// ============================================================
static int tensor_count(int order) {
    int c = 0, d = 1;
    for (int n = 0; n <= order; n++) { c += d; d *= DIM; }
    return c;
}

struct TensorCoeffs {
    static const int MAX_COEFFS = 1 + 3 + 9 + 27 + 81 + 243; // 支持到5阶
    double k[MAX_COEFFS];   // 系数存储数组
    int Order;              // 当前使用的最高阶数

    TensorCoeffs() : Order(0) { std::memset(k, 0, sizeof(k)); }
    TensorCoeffs(int order, double val) : Order(order) {
        int nc = tensor_count(order);
        for (int i = 0; i < nc; i++) k[i] = val;
        for (int i = nc; i < MAX_COEFFS; i++) k[i] = 0;
    }

    int ncoeffs() const { return tensor_count(Order); }

    // 累加: this += other (插值时用)
    void operator+=(const TensorCoeffs& o) {
        int nc = ncoeffs();
        for (int i = 0; i < nc; i++) k[i] += o.k[i];
    }
    // 标量乘: this *= s (Lagrange权重乘法)
    void operator*=(double s) {
        int nc = ncoeffs();
        for (int i = 0; i < nc; i++) k[i] *= s;
    }

    // 访问第ni阶、索引为abc[0..ni-1]的张量分量
    // 内存布局: [0阶(1个)] [1阶(3个)] [2阶(9个)] [3阶(27个)] ...
    double& get(int ni, const int abc[]) {
        if (ni == 0) return k[0];
        if (ni == 1) return k[1 + abc[0]];
        if (ni == 2) return k[1+DIM + abc[0] + abc[1]*DIM];
        if (ni == 3) return k[1+DIM+DIM*DIM + abc[0] + abc[1]*DIM + abc[2]*DIM*DIM];
        if (ni == 4) return k[1+DIM+DIM*DIM+DIM*DIM*DIM + abc[0]+abc[1]*DIM+abc[2]*DIM*DIM+abc[3]*DIM*DIM*DIM];
        assert(false);
        return k[0];
    }
};

// ============================================================
//  从分布函数f计算Hermite张量系数 (正变换: f → a_n)
//
//  a_n^{abc...} = Σ_{iq} H_n(v_iq; abc...) * f[iq]
//
//  其中 v_iq = ef[iq]*gw + gu 是gauge变换后的实际速度。
//
//  gauge的含义:
//    gu = 宏观速度 (使离散速度中心化)
//    gw = sqrt(T/TLat) (使离散速度按温度缩放)
//    实际速度: v_iq = ef[iq]*gw + gu
//
//  这样做的目的是让每个格点的分布函数在其自身的
//  局部参考系(co-moving frame)中表示, 使得张量系数
//  在空间上变化平缓, 有利于插值精度。
//
//  对应原始代码: momentsMatrix.cuh :: calc_moments_tensors()
// ============================================================
static void calc_moments_tensors(const double gu[3], double gw,
                                 const double f[QN], TensorCoeffs& TC) {
    // 遍历每一阶 (0阶=标量, 1阶=向量, 2阶=张量, ...)
    for (int ni = 0; ni <= TC.Order; ni++) {
        // Ndim = DIM^ni: 该阶的独立分量数 (DIM=3: 1,3,9,27,...)
        int Ndim = 1;
        for (int i = 0; i < ni; i++) Ndim *= DIM;

        // 遍历该阶的所有张量分量
        for (int abc = 0; abc < Ndim; abc++) {
            // 将线性索引abc解码为多维索引 abci[0..ni-1]
            // 例如 ni=2, abc=5: abci = {5%3, 5/3%3} = {2, 1} 即 (z,y)分量
            int abci[5] = {0,0,0,0,0};
            for (int i = 0, Nds = 1; i < ni; i++, Nds *= DIM)
                abci[i] = (abc / Nds) % DIM;

            double& an = TC.get(ni, abci);
            an = 0;
            // 对所有27个离散方向求和: a_n = Σ H_n(v) * f
            for (int iq = 0; iq < QN; iq++) {
                // 计算gauge变换后的实际速度
                double v[3] = {ef[iq].x*gw + gu[0],
                               ef[iq].y*gw + gu[1],
                               ef[iq].z*gw + gu[2]};
                an += Hermite(ni, v, abci) * f[iq];
            }
        }
    }
}

// ============================================================
//  A系数 → D系数的转换 (从局部运动参考系到格子参考系)
//
//  A系数 (a_n): 在源格点的局部gauge下计算的Hermite矩
//  D系数 (d_n): 在目标格点的gauge下的标准Hermite系数
//
//  转换关系 (以0~2阶为例):
//    d_0 = a_0                                    (密度不变)
//    d_1_a = (1/√T) * (a_1_a - u_a * a_0)         (减去整体动量)
//    d_2_ab = (1/T) * (a_2_ab - u_a*A1_b - u_b*A1_a
//                     + (u_a*u_b - (T-1)*δ_ab)*a_0) (减去速度和温度贡献)
//
//  其中 √T = gw = sqrt(T/TLat), u = gauge速度。
//
//  这个转换本质上是一个Galilean变换 + 温度缩放:
//  将物理空间的矩转换为格子空间（标准化）的矩。
//
//  对应原始代码: momentsMatrix.cuh :: convertAtoD()
// ============================================================
static TensorCoeffs convertAtoD(TensorCoeffs& an, const double gu[3], double gw) {
    const double sqT  = gw;           // √(T/TLat)
    const double dsqT = 1.0 / sqT;    // 1/√(T/TLat)
    const double T    = sqT * sqT;     // T/TLat
    const double u[3] = {gu[0], gu[1], gu[2]};  // gauge速度

    TensorCoeffs dn;
    dn.Order = an.Order;

    // 获取0阶系数 a_0 (=密度ρ)
    int z5[5] = {0,0,0,0,0};
    const double an0 = an.get(0, z5);

    for (int ni = 0; ni <= an.Order; ni++) {
        int Ndim = 1;
        for (int i = 0; i < ni; i++) Ndim *= DIM;

        for (int abc = 0; abc < Ndim; abc++) {
            int abci[5] = {0,0,0,0,0};
            for (int i = 0, Nds = 1; i < ni; i++, Nds *= DIM)
                abci[i] = (abc / Nds) % DIM;

            const int a = abci[0], b = abci[1], c = abci[2];
            double& dn_val = dn.get(ni, abci);
            double  an_val = an.get(ni, abci);

            if (ni == 0) {
                // 0阶: d_0 = a_0 (密度不变)
                dn_val = an_val;
            } else if (ni == 1) {
                // 1阶: d_1_a = (1/√T) * (a_1_a - u_a * ρ)
                // 物理意义: 减去背景流速的贡献
                dn_val = dsqT * (an_val - u[a] * an0);
            } else if (ni == 2) {
                // 2阶: d_2_ab = (1/T) * (a_2_ab - u_a*A1_b - u_b*A1_a + ...)
                // 物理意义: 提取纯应力张量（去除速度和温度的贡献）
                int ia[5] = {a,0,0,0,0}, ib[5] = {b,0,0,0,0};
                double An1_a = an.get(1, ia);  // a_1_a (a方向的一阶矩)
                double An1_b = an.get(1, ib);  // a_1_b (b方向的一阶矩)
                dn_val = dsqT * dsqT * (
                    an_val
                    - u[a] * An1_b - u[b] * An1_a
                    + (u[a]*u[b] - (T-1)*(a==b)) * an0
                );
            } else if (ni == 3) {
                // 3阶转换 (类似但更复杂)
                int ia[5]={a}, ib[5]={b}, ic[5]={c};
                int ibc[5]={b,c}, iac[5]={a,c}, iab[5]={a,b};
                double An1_a = an.get(1,ia), An1_b = an.get(1,ib), An1_c = an.get(1,ic);
                double An2_bc = an.get(2,ibc), An2_ac = an.get(2,iac), An2_ab = an.get(2,iab);
                double u1_ab = u[a]*u[b]-(T-1)*(a==b);
                double u1_ac = u[a]*u[c]-(T-1)*(a==c);
                double u1_bc = u[b]*u[c]-(T-1)*(b==c);
                dn_val = dsqT*dsqT*dsqT * (
                    an_val
                    - (u[a]*An2_bc + u[b]*An2_ac + u[c]*An2_ab)
                    + (u1_ab*An1_c + u1_ac*An1_b + u1_bc*An1_a)
                    - (u[a]*u[b]*u[c] - (T-1)*(u[a]*(b==c)+u[b]*(a==c)+u[c]*(a==b))) * an0
                );
            } else if (ni == 4) {
                // 4阶转换 (最复杂, 包含大量交叉项)
                const int d = abci[3];
                int ia[5]={a}, ib[5]={b}, ic[5]={c}, id[5]={d};
                int iab[5]={a,b}, iac[5]={a,c}, iad[5]={a,d};
                int ibc[5]={b,c}, ibd[5]={b,d}, icd[5]={c,d};
                int ibcd[5]={b,c,d}, iacd[5]={a,c,d}, iabd[5]={a,b,d}, iabc[5]={a,b,c};
                double An1_a=an.get(1,ia), An1_b=an.get(1,ib), An1_c=an.get(1,ic), An1_d=an.get(1,id);
                double An2_ab=an.get(2,iab), An2_ac=an.get(2,iac), An2_ad=an.get(2,iad);
                double An2_bc=an.get(2,ibc), An2_bd=an.get(2,ibd), An2_cd=an.get(2,icd);
                double An3_bcd=an.get(3,ibcd), An3_acd=an.get(3,iacd);
                double An3_abd=an.get(3,iabd), An3_abc=an.get(3,iabc);
                double u1_ab=u[a]*u[b]-(T-1)*(a==b), u1_ac=u[a]*u[c]-(T-1)*(a==c);
                double u1_ad=u[a]*u[d]-(T-1)*(a==d), u1_bc=u[b]*u[c]-(T-1)*(b==c);
                double u1_bd=u[b]*u[d]-(T-1)*(b==d), u1_cd=u[c]*u[d]-(T-1)*(c==d);
                dn_val = dsqT*dsqT*dsqT*dsqT * (
                    an_val
                    -(u[a]*An3_bcd + u[b]*An3_acd + u[c]*An3_abd + u[d]*An3_abc)
                    +(u1_ab*An2_cd + u1_ac*An2_bd + u1_ad*An2_bc + u1_bc*An2_ad + u1_bd*An2_ac + u1_cd*An2_ab)
                    -(u[a]*u[b]*u[c]*An1_d + u[a]*u[b]*u[d]*An1_c + u[a]*u[c]*u[d]*An1_b + u[b]*u[c]*u[d]*An1_a)
                    +(T-1)*(u[a]*(b==c)*An1_d+u[a]*(b==d)*An1_c+u[a]*(c==d)*An1_b
                           +u[b]*(c==d)*An1_a+u[b]*(a==d)*An1_c+u[b]*(a==c)*An1_d
                           +u[c]*(b==d)*An1_a+u[c]*(a==d)*An1_b+u[c]*(a==b)*An1_d
                           +u[d]*(b==c)*An1_a+u[d]*(a==c)*An1_b+u[d]*(a==b)*An1_c)
                    + u[a]*u[b]*u[c]*u[d]*an0
                    -(T-1)*(u[a]*u[b]*(c==d)+u[a]*u[c]*(b==d)+u[a]*u[d]*(b==c)
                           +u[b]*u[c]*(a==d)+u[b]*u[d]*(a==c)+u[c]*u[d]*(a==b))*an0
                    +(T-1)*(T-1)*((a==b)*(c==d)+(a==c)*(b==d)+(a==d)*(b==c))*an0
                );
            } else {
                assert(false && "RegOrder > 4 not implemented");
            }
        }
    }
    return dn;
}

// ============================================================
//  从D系数重构分布函数 (逆变换: d_n → f_i)
//
//  f_i = Σ_{n=0}^{Order} Σ_{abc} (w_i / n!) * d_n^{abc} * H_n(ef_i; abc)
//
//  这是Hermite展开的求值公式:
//  将标准Hermite系数和格子方向上的Hermite多项式加权求和,
//  乘以格子权重w_i, 得到该方向上的分布函数值。
//
//  注意: 这里使用的是格子参考系的速度ef[iq] (不加gauge偏移),
//  因为D系数已经是在格子参考系中了。
//
//  对应原始代码: momentsMatrix.cuh :: eval_fi_Hermit()
// ============================================================
static double eval_fi_Hermit(TensorCoeffs& TC, int iq) {
    double fi = 0;
    const double ev[3] = {ef[iq].x, ef[iq].y, ef[iq].z};  // 格子速度方向
    for (int ni = 0; ni <= TC.Order; ni++) {
        int Ndim = 1, nfact = 1;
        // 计算 DIM^ni 和 ni!
        for (int i = 1; i <= ni; i++) { Ndim *= DIM; nfact *= i; }

        for (int abc = 0; abc < Ndim; abc++) {
            int abci[5] = {0,0,0,0,0};
            for (int i = 0, Nds = 1; i < ni; i++, Nds *= DIM)
                abci[i] = (abc / Nds) % DIM;

            double dn = TC.get(ni, abci);
            // 贡献 = w_i * (1/n!) * d_n * H_n(ef_i)
            fi += wt[iq] / nfact * dn * Hermite(ni, ev, abci);
        }
    }
    return fi;
}

// ============================================================
//  矩矩阵 (用于 RegOrder < 0 的全矩阵求逆方法)
//
//  构建 27×27 的矩矩阵 M, 其中:
//    M[i][j] = v_j^{px_i} * v_j^{py_i} * v_j^{pz_i}
//
//  然后通过Gauss-Jordan消元求逆, 得到 M^{-1}。
//
//  分布函数的重构:
//    先计算矩向量 mom = M_source * f (在源gauge下)
//    再重构: f_i = Σ_j M^{-1}_target[i][j] * mom[j]
//
//  这个方法保留了所有阶的矩（不做截断），精度最高，
//  但需要对27×27矩阵求逆, 计算开销大。
//
//  对应原始代码: momentsMatrix.cuh :: MomentsMatrix
// ============================================================
struct MomentsMatrix {
    double m[QN][QN * 2];   // 增广矩阵 [M | I], 求逆后变为 [I | M^{-1}]

    // 构建矩矩阵: M[i][j] = Π_d v_j[d]^{power_i[d]}
    // gauge: gu=速度, gw=温度缩放因子
    void init(const double gu[3], double gw) {
        for (int i = 0; i < QN; i++)        // 行: 第i个矩 (幂次由MomentsPower[i]定义)
            for (int j = 0; j < QN; j++) {   // 列: 第j个速度方向
                // 计算gauge变换后的实际速度 v_j
                double v[3] = {ef[j].x*gw + gu[0],
                               ef[j].y*gw + gu[1],
                               ef[j].z*gw + gu[2]};
                // M[i][j] = v_j_x^{px} * v_j_y^{py} * v_j_z^{pz}
                m[i][j] = ipow(v[0], MomentsPower[i][0])
                         * ipow(v[1], MomentsPower[i][1])
                         * ipow(v[2], MomentsPower[i][2]);
            }
    }

    // Gauss-Jordan就地求逆
    // 将增广矩阵 [M | I] 变换为 [I | M^{-1}]
    void inverse() {
        // 初始化右半部分为单位矩阵
        for (int i = 0; i < QN; i++)
            for (int j = QN; j < 2*QN; j++)
                m[i][j] = (i == j - QN) ? 1.0 : 0.0;

        for (int i = 0; i < QN; i++) {
            // 列主元选取: 找第i列绝对值最大的行作为主元
            int pivot = i;
            for (int j = i+1; j < QN; j++)
                if (std::fabs(m[j][i]) > std::fabs(m[pivot][i])) pivot = j;

            // 交换主元行
            if (pivot != i)
                for (int k = 0; k < 2*QN; k++) std::swap(m[i][k], m[pivot][k]);

            // 消元: 用第i行消去其他所有行的第i列
            for (int j = 0; j < QN; j++) {
                if (i != j) {
                    double ratio = m[j][i] / m[i][i];
                    for (int k = 0; k < 2*QN; k++) m[j][k] -= ratio * m[i][k];
                } else {
                    // 主元行归一化
                    double div = 1.0 / m[i][i];
                    for (int k = 0; k < 2*QN; k++) m[i][k] *= div;
                }
            }
        }
    }

    // 用逆矩阵重构分布函数: f_i = Σ_j M^{-1}[i][j] * mom[j]
    double get_inv(int irow, const double mvec[QN]) const {
        double fi = 0;
        for (int j = 0; j < QN; j++) fi += m[irow][j + QN] * mvec[j];
        return fi;
    }
};

// ============================================================
//  计算矩向量 (用于 RegOrder < 0 的全矩阵方法)
//
//  mom[i] = Σ_j v_j^{p_i} * f[j]
//
//  其中 p_i = MomentsPower[i] 定义了第i个矩的幂次,
//  v_j = ef[j]*gw + gu 是gauge变换后的速度。
//
//  物理意义: 第i个矩是分布函数对速度多项式的加权求和。
//  例如 p=(0,0,0)→密度, p=(1,0,0)→x动量, p=(2,0,0)→x方向二阶矩...
//
//  对应原始代码: momentsMatrix.cuh :: calc_moments_vec()
// ============================================================
static void calc_moments_vec(const double gu[3], double gw,
                             const double f[QN], double mom[QN]) {
    for (int i = 0; i < QN; i++) {
        mom[i] = 0;
        for (int j = 0; j < QN; j++) {
            double v[3] = {ef[j].x*gw + gu[0],
                           ef[j].y*gw + gu[1],
                           ef[j].z*gw + gu[2]};
            mom[i] += ipow(v[0], MomentsPower[i][0])
                    * ipow(v[1], MomentsPower[i][1])
                    * ipow(v[2], MomentsPower[i][2]) * f[j];
        }
    }
}

// ============================================================
//  初始化: 剪切波
//
//  剪切波是LBM的经典验证用例, 用于检验粘度精度。
//
//  初始条件:
//    - 密度 ρ = 1 (均匀)
//    - 温度 T = T0 (均匀)
//    - 速度: 正弦形式的横向速度扰动 + 均匀背景流速(uDrag)
//
//  对于 shearWaveDir=3 (XYZ对角方向):
//    u_l = u0 * sin(2π(ix+iy+iz)/Nx)
//    vx = -u_l/√6 + uDragX
//    vy = -u_l/√6 + uDragY
//    vz = +u_l*√(2/3) + uDragZ
//
//  波矢量沿(1,1,1)方向, 速度扰动垂直于波矢量方向。
//  剪切波的衰减率应为 exp(-ν*k²*t), 其中 k=2π/Nx, ν=粘度。
//
//  分布函数初始化为对应宏观量的平衡态: f = f^eq(ρ, vel, T)。
//
//  对应原始代码: materials.cuh :: shear_wave()
// ============================================================
static void init_shear_wave() {
    // 分配双缓冲内存
    cells[0].resize(Nx * Ny * Nz);
    cells[1].resize(Nx * Ny * Nz);

    for (int iz = 0; iz < Nz; iz++)
    for (int iy = 0; iy < Ny; iy++)
    for (int ix = 0; ix < Nx; ix++) {
        double vx = 0, vy = 0, vz = 0;

        if (PP.shearWaveDir == 1) {
            // 1D剪切波: 沿x方向传播
            vx = PP.u0 * std::sin(2*M_PI * ix / Nx);
        } else if (PP.shearWaveDir == 2) {
            // 2D剪切波: 沿(1,1,0)对角方向传播
            double u_l = PP.u0 * std::sin(2*M_PI * (ix+iy) / Nx);
            vx = -u_l / std::sqrt(2.0);
            vy =  u_l / std::sqrt(2.0);
        } else if (PP.shearWaveDir == 3) {
            // 3D剪切波: 沿(1,1,1)对角方向传播
            double u_l = PP.u0 * std::sin(2*M_PI * (ix+iy+iz) / Nx);
            vx = -u_l / std::sqrt(6.0);
            vy = -u_l / std::sqrt(6.0);
            vz =  u_l * std::sqrt(2.0 / 3.0);
        }
        // 加上均匀背景流速
        vx += PP.uDragX;
        vy += PP.uDragY;
        vz += PP.uDragZ;

        Cell& c   = cells[0][idx(ix, iy, iz)];
        c.rho = 1.0;          // 均匀密度
        c.vel = {vx, vy, vz}; // 初始速度
        c.T   = PP.T0;        // 均匀温度

        // 用平衡态初始化分布函数: f = feq(ρ, vel, T)
        double feq[QN];
        calcEq(feq, c.rho, c.vel, c.T);
        for (int iq = 0; iq < QN; iq++) c.f[iq] = feq[iq];

        // 复制到第二个缓冲区
        cells[1][idx(ix, iy, iz)] = c;
    }
}

// ============================================================
//  POD半拉格朗日流步 + BGK碰撞 (一个时间步)
//
//  这是程序的核心函数, 实现POD半拉格朗日LBM的完整时间推进。
//
//  与传统LBM的关键区别:
//  1. 传统LBM: 粒子沿整数格点方向传播 → 精确到达相邻格点 → 无需插值
//  2. POD LBM: 粒子沿浮点方向传播 → 出发点不在格点上 → 需要插值重构
//
//  流步算法:
//  对每个格点(ix,iy,iz):
//    Step 1: 确定gauge (标度参数)
//      gauge = (vel, sqrt(T/TLat))
//      作用: 将格子速度缩放到物理速度空间
//
//    Step 2: 对每个离散方向iq, 执行半拉格朗日回溯:
//      a. 计算实际速度: v_iq = ef[iq]*sqrt(T/TLat) + vel
//      b. 求出发点: xf = (ix,iy,iz) - v_iq (特征线回溯)
//      c. 确定Lagrange插值模板 (出发点周围的格点集合)
//      d. 在矩空间(而非分布函数空间)进行插值:
//         - RegOrder >= 0: Hermite张量正则化
//           * 在每个模板点计算Hermite张量系数
//           * Lagrange插值这些系数
//           * 用convertAtoD转换到目标gauge
//           * 用eval_fi_Hermit重构f
//         - RegOrder < 0: 全矩阵求逆
//           * 构建目标gauge的矩矩阵并求逆
//           * 在每个模板点计算矩向量
//           * Lagrange插值矩向量
//           * 用逆矩阵重构f
//
//    Step 3: 迭代检查
//      从新的f计算宏观量 → 更新gauge → 重复Step 2
//      直到收敛（因为gauge依赖宏观量, 而宏观量依赖插值结果）
//
//    Step 4: 碰撞
//      f_new = f - (1/τ)(f - feq(ρ, 0, TLat))
//      注意: 平衡态用零速度! 因为半拉格朗日步已处理了对流。
//
//  对应原始代码: streaming-pod.cu :: streaming_collision<>()
// ============================================================
static void streaming_collision_step() {
    const int ild = 0;   // 读缓冲区索引 (旧时刻数据)
    const int ist = 1;   // 写缓冲区索引 (新时刻数据)
    const int Npoints = PP.stencilInterpWidth + 1;  // 每个方向的插值点数

    // 遍历所有格点
    for (int iz = 0; iz < Nz; iz++)
    for (int iy = 0; iy < Ny; iy++)
    for (int ix = 0; ix < Nx; ix++) {
        // 从旧缓冲区读取当前格点数据作为迭代初值
        Cell cell = cells[ild][idx(ix, iy, iz)];
        Cell cell_new;

        // ======== 内迭代 (定点迭代法) ========
        // 最多迭代100次, 通常2~5次即可收敛
        int Niter = 0;
        while (Niter < 100) {
            // 计算当前gauge:
            //   gw = sqrt(T/TLat): 温度缩放因子, 使离散速度适应当前温度
            //   gu = vel: 整体流速, 使离散速度中心化到运动参考系
            const double gw = std::sqrt(cell.T / TLat);
            const double gu[3] = {cell.vel.x, cell.vel.y, cell.vel.z};

            // 全矩阵方法: 构建并求逆目标gauge下的矩矩阵 (每次迭代重算一次)
            MomentsMatrix Mm;
            if (PP.RegOrder < 0) {
                Mm.init(gu, gw);
                Mm.inverse();
            }

            // ======== 对每个离散速度方向进行半拉格朗日回溯 ========
            for (int iq = 0; iq < QN; iq++) {
                // 计算第iq个方向在实验室坐标系中的实际速度
                // v = ef[iq] * sqrt(T/TLat) + vel
                Vec3 v = {ef[iq].x * gw + gu[0],
                          ef[iq].y * gw + gu[1],
                          ef[iq].z * gw + gu[2]};

                // 特征线回溯: 出发点 = 当前位置 - 速度*dt (dt=1)
                Vec3 xf = {ix - v.x, iy - v.y, iz - v.z};

                // 确定插值模板原点
                int sMinX, sMinY, sMinZ;
                if (PP.stencilFixed) {
                    // 固定模板: 始终以当前格点为中心
                    sMinX = ix - PP.stencilInterpWidth / 2;
                    sMinY = iy - PP.stencilInterpWidth / 2;
                    sMinZ = iz - PP.stencilInterpWidth / 2;
                } else {
                    // 自适应模板: 将模板中心对准出发点 (精度更高)
                    double hw = 0.5 * PP.stencilInterpWidth;
                    sMinX = (int)std::round(xf.x - hw);
                    sMinY = (int)std::round(xf.y - hw);
                    sMinZ = (int)std::round(xf.z - hw);
                }

                // 出发点在模板中的分数坐标 (用于Lagrange插值)
                Vec3 shifts = {xf.x - sMinX, xf.y - sMinY, xf.z - sMinZ};

                if (PP.RegOrder < 0) {
                    // ===== 方法一: 全矩阵求逆 (RegOrder < 0) =====
                    //
                    // 算法:
                    // 1. 在每个模板点, 用其局部gauge计算矩向量
                    // 2. 用Lagrange权重加权这些矩向量
                    // 3. 用目标gauge的逆矩阵将插值后的矩还原为f
                    //
                    // 优点: 保留所有阶矩, 无截断误差
                    // 缺点: 需要27×27矩阵求逆, 计算量大
                    double val = 0;
                    for (int xs = 0; xs < Npoints; xs++)
                    for (int ys = 0; ys < Npoints; ys++)
                    for (int zs = 0; zs < Npoints; zs++) {
                        // 模板点的周期性坐标
                        int cx = wrap(sMinX + xs, Nx);
                        int cy = wrap(sMinY + ys, Ny);
                        int cz = wrap(sMinZ + zs, Nz);
                        const Cell& sc = cells[ild][idx(cx, cy, cz)];

                        // Lagrange插值权重
                        double coeff = LagrPol(xs, ys, zs, shifts, Npoints);

                        // 源格点的gauge
                        double ig[3] = {sc.vel.x, sc.vel.y, sc.vel.z};
                        double igw   = std::sqrt(sc.T / TLat);

                        // 在源gauge下计算矩向量
                        double mVec[QN];
                        calc_moments_vec(ig, igw, sc.f, mVec);

                        // 用目标gauge的逆矩阵将矩还原为f, 乘以插值权重
                        val += coeff * Mm.get_inv(iq, mVec);
                    }
                    cell_new.f[iq] = val;
                } else {
                    // ===== 方法二: Hermite张量正则化 (RegOrder >= 0) =====
                    //
                    // 算法:
                    // 1. 在每个模板点, 用其局部gauge计算Hermite张量系数 (A系数)
                    // 2. 用Lagrange权重加权插值这些张量系数
                    // 3. 用convertAtoD将插值后的A系数转换为目标gauge的D系数
                    // 4. 用eval_fi_Hermit从D系数重构f
                    //
                    // 优点: 只保留低阶矩, 过滤高频噪声 (正则化效果)
                    // 缺点: 截断高阶矩会引入误差
                    //
                    // RegOrder=2: 保留0+1+2阶 = 密度+动量+应力张量 (13个系数)
                    TensorCoeffs an_interp(PP.RegOrder, 0.0);

                    for (int xs = 0; xs < Npoints; xs++)
                    for (int ys = 0; ys < Npoints; ys++)
                    for (int zs = 0; zs < Npoints; zs++) {
                        int cx = wrap(sMinX + xs, Nx);
                        int cy = wrap(sMinY + ys, Ny);
                        int cz = wrap(sMinZ + zs, Nz);
                        const Cell& sc = cells[ild][idx(cx, cy, cz)];

                        double coeff = LagrPol(xs, ys, zs, shifts, Npoints);

                        // 源格点的gauge
                        double ig[3] = {sc.vel.x, sc.vel.y, sc.vel.z};
                        double igw   = std::sqrt(sc.T / TLat);

                        // 计算源格点的Hermite张量系数 (A系数, 在源gauge下)
                        TensorCoeffs an_p(PP.RegOrder, 0.0);
                        calc_moments_tensors(ig, igw, sc.f, an_p);
                        // 乘以Lagrange权重并累加
                        an_p *= coeff;
                        an_interp += an_p;
                    }

                    // 将插值后的A系数转换为目标gauge下的D系数
                    TensorCoeffs dn = convertAtoD(an_interp, gu, gw);
                    // 从D系数重构分布函数
                    cell_new.f[iq] = eval_fi_Hermit(dn, iq);
                }
            }

            // ======== 从新的f计算宏观量 ========
            // ρ = Σ f_i (零阶矩 = 密度)
            // ρu = Σ v_i * f_i (一阶矩 = 动量)
            // M2 = Σ |v_i|² * f_i (二阶矩迹 → 温度)
            double sum_rho = 0, sum_vx = 0, sum_vy = 0, sum_vz = 0, sum_M2 = 0;
            for (int ik = 0; ik < QN; ik++) {
                Vec3 v_k = {ef[ik].x*gw + gu[0],
                            ef[ik].y*gw + gu[1],
                            ef[ik].z*gw + gu[2]};
                sum_rho += cell_new.f[ik];           // 密度
                sum_vx  += v_k.x * cell_new.f[ik];  // x方向动量
                sum_vy  += v_k.y * cell_new.f[ik];  // y方向动量
                sum_vz  += v_k.z * cell_new.f[ik];  // z方向动量
                sum_M2  += dot(v_k, v_k) * cell_new.f[ik];  // 二阶矩迹
            }
            cell_new.rho = sum_rho;
            cell_new.vel = {sum_vx/sum_rho, sum_vy/sum_rho, sum_vz/sum_rho};
            // 温度 = (二阶矩迹/ρ - |vel|²) / DIM
            // 即从总动能中减去整体动能, 得到热运动能, 再除以维度
            cell_new.T   = sum_M2/sum_rho - dot(cell_new.vel, cell_new.vel);
            cell_new.T  /= DIM;

            // 固定温度模式: 温度不随时间演化 (用于纯剪切波测试)
            if (PP.fixedTemperature) cell_new.T = cell.T;
            // 安全检查: 温度不应为负
            if (cell_new.T < 0) {
                printf("Warning: negative T at (%d,%d,%d) iter %d, T=%g\n",
                       ix, iy, iz, Niter, cell_new.T);
                cell_new.T = -cell_new.T;
            }

            Niter++;
            // 检查收敛: 如果新旧宏观量足够接近, 停止迭代
            if (isConv(cell, cell_new)) { cell = cell_new; break; }
            cell = cell_new;  // 用新值作为下一次迭代的初值
        }

        // ======== BGK碰撞 ========
        // POD方法中, 平衡态用零速度和格子温度计算:
        //   feq = feq(ρ, u=0, T=TLat)
        // 原因: 半拉格朗日步已经处理了对流(平流),
        // 碰撞只需要在格子静止参考系中进行松弛。
        //
        // 对于零速度+TLat, feq简化为: feq_i = w_i * ρ
        double feq[QN];
        calcEq(feq, cell.rho, {0, 0, 0}, TLat);
        collision(cell.f, feq);

        // 将碰撞后的结果写入新缓冲区
        cells[ist][idx(ix, iy, iz)] = cell;
    }

    // 交换双缓冲: 新数据变为旧数据, 为下一步做准备
    std::swap(cells[0], cells[1]);
}

// ============================================================
//  诊断输出: 打印守恒量
//  - 质量 (应守恒, 但半拉格朗日方法可能有微小漂移)
//  - 动量 (如果没有外力, 也应守恒)
//  - 最大速度 (监控稳定性)
// ============================================================
static void print_diagnostics(int step) {
    double mass = 0, maxVel = 0;
    Vec3 mom = {0, 0, 0};

    for (int i = 0; i < Nx*Ny*Nz; i++) {
        const Cell& c = cells[0][i];
        mass  += c.rho;
        mom.x += c.rho * c.vel.x;
        mom.y += c.rho * c.vel.y;
        mom.z += c.rho * c.vel.z;
        double vmag = std::sqrt(dot(c.vel, c.vel));
        if (vmag > maxVel) maxVel = vmag;
    }
    printf("Step %6d | Mass %.12f Momentum(%.9f %.9f %.9f) MaxVelocity %.12f\n",
           step, mass, mom.x, mom.y, mom.z, maxVel);
}

// ============================================================
//  主函数
//
//  参数设置来自 shear_wave_test.py:
//    - 网格: 默认20³ (CPU快速测试; 原始测试用100³)
//    - 粘度: visc=0.2
//    - Ma数: Ma=10 (背景流速/声速)
//    - 剪切波方向: 3 (XYZ对角)
//    - 温度: T0=1/3, 固定温度模式
//    - 等温松弛, 4阶平衡态展开
//    - RegOrder=2 (2阶Hermite正则化)
//    - 2阶Lagrange插值 (3点模板)
// ============================================================
int main(int argc, char** argv) {
    // 默认参数 (可通过命令行覆盖)
    Nx = 20;  Ny = 20;  Nz = 20;
    int MaxSteps = 100;
    int RegOrder = 2;

    // 解析命令行参数
    if (argc > 1) Nx = std::atoi(argv[1]);
    if (argc > 2) Ny = std::atoi(argv[2]);
    if (argc > 3) Nz = std::atoi(argv[3]);
    if (argc > 4) MaxSteps = std::atoi(argv[4]);
    if (argc > 5) RegOrder = std::atoi(argv[5]);

    printf("=== POD Semi-Lagrangian LBM — Shear Wave Test (CPU) ===\n");
    printf("Grid: %d x %d x %d,  Steps: %d,  RegOrder: %d\n",
           Nx, Ny, Nz, MaxSteps, RegOrder);
    printf("Lattice: D3Q27 POD (TLat=%.1f, ec=sqrt(3)=%.6f)\n\n", TLat, EC);

    // ---- 设置物理参数 ----
    PP.visc_atT            = 0.2;   // 粘度 (格子单位)
    PP.dr                  = 1.0;   // 空间步长
    PP.dt                  = 1.0;   // 时间步长
    PP.stencilInterpWidth  = 2;     // Lagrange插值宽度 (使用3点)
    PP.stencilFixed        = 0;     // 自适应模板 (=1则固定模板)
    PP.RegOrder            = RegOrder;
    PP.EquilibriumOrder    = 4;     // 平衡态展开到4阶
    PP.IsothermalRelaxation= 1;     // 等温松弛
    PP.fixedTemperature    = 1;     // 温度不演化
    PP.MaxSteps            = MaxSteps;
    PP.StepIterPeriod      = 1;     // 每步都输出

    // ---- 剪切波参数 ----
    const double Tinit = 1.0 / 3.0;   // 初始温度 (传统LBM标准温度)
    const double Ma_a  = 10.0;        // 声学Mach数 (背景流速/声速)
    const int    waveD = 3;            // 剪切波方向 (3=XYZ对角)

    PP.T0           = Tinit;
    PP.u0           = 0.05;   // 剪切波速度振幅
    PP.rho0         = 1.0;    // 初始密度
    PP.shearWaveDir = waveD;
    // 背景拖拽速度: uDrag = Ma * sqrt(T0) / sqrt(waveD)
    // 使得在每个方向上的Mach数相同
    PP.uDragX = (Ma_a / std::sqrt((double)waveD)) * std::sqrt(Tinit);
    PP.uDragY = (waveD > 1) ? PP.uDragX : 0;
    PP.uDragZ = (waveD > 2) ? PP.uDragX : 0;
    PP.setupUnits();  // 计算τ和1/τ

    printf("  visc=%.4f  tau=%.6f  dtau=%.6f\n", PP.visc_atT, PP.tau, PP.dtau);
    printf("  T0=%.6f  u0=%.4f  Ma=%.1f  waveDir=%d\n", PP.T0, PP.u0, Ma_a, waveD);
    printf("  uDrag=(%.6f, %.6f, %.6f)\n\n", PP.uDragX, PP.uDragY, PP.uDragZ);

    // 初始化剪切波场
    init_shear_wave();
    print_diagnostics(0);

    // ---- 时间推进主循环 ----
    for (int step = 1; step <= MaxSteps; step++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 执行一步: 半拉格朗日流步 + BGK碰撞
        streaming_collision_step();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (step % PP.StepIterPeriod == 0) {
            print_diagnostics(step);
            // MLU/s = 百万格点更新每秒 (性能指标)
            printf("  (%.1f ms, %.3f MLU/s)\n", ms, 1e-3*Nx*Ny*Nz / ms);
        }
    }

    printf("\nDone.\n");
    return 0;
}
