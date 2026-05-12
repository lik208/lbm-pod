// lbm_pod_cpu.cpp
// ====================================================================
// Simplified CPU-only POD (Semi-Lagrangian) LBM solver
// for shear wave decay test using D3Q27 lattice.
//
// Based on: https://github.com/zakirovandrey/lbm-pod
//
// Compile:
//   g++ -O2 -std=c++14 -o lbm_pod_cpu lbm_pod_cpu.cpp -lm
//
// Run:
//   ./lbm_pod_cpu [Nx] [Ny] [Nz] [MaxSteps] [RegOrder]
//
// Default: Nx=Ny=Nz=20, MaxSteps=100, RegOrder=2
// RegOrder: -1 = full matrix inverse, >=0 = Hermite tensor regularization
// ====================================================================

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <chrono>

// ============================================================
//  Vector3 type (replaces CUDA float3/double3)
// ============================================================
struct Vec3 {
    double x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator*(double s, Vec3 a) { return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator/(Vec3 a, double s) { return {a.x/s, a.y/s, a.z/s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

// ============================================================
//  D3Q27 POD Lattice Constants
// ============================================================
static const int DIM = 3;
static const int QN  = 27;

static const double TLat = 1.0;
static const double cs2  = TLat;
static const double dcs2 = 1.0 / cs2;

// Discrete velocities (floating-point, not integer — key feature of POD)
// Each component is 0, ±sqrt(3) depending on the shell
static const double EC = std::sqrt(3.0);

static const Vec3 ef[QN] = {
    { 0,   0,   0  },
    {+EC,  0,   0  }, {-EC,  0,   0  }, { 0,  +EC,  0  }, { 0,  -EC,  0  }, { 0,   0,  +EC }, { 0,   0,  -EC },
    {+EC, +EC,  0  }, {+EC,  0,  +EC }, { 0,  +EC, +EC },
    {+EC, -EC,  0  }, {+EC,  0,  -EC }, { 0,  +EC, -EC },
    {-EC, +EC,  0  }, {-EC,  0,  +EC }, { 0,  -EC, +EC },
    {-EC, -EC,  0  }, {-EC,  0,  -EC }, { 0,  -EC, -EC },
    {+EC, +EC, +EC }, {-EC, +EC, +EC }, {+EC, -EC, +EC }, {+EC, +EC, -EC },
    {-EC, -EC, +EC }, {-EC, +EC, -EC }, {+EC, -EC, -EC }, {-EC, -EC, -EC }
};

// Weights: W0 (rest), W1 (face), W2 (edge), W3 (corner)
static const double W0 = 8.0 / 27.0;
static const double W1 = 2.0 / 27.0;
static const double W2 = 1.0 / 54.0;
static const double W3 = 1.0 / 216.0;

static const double wt[QN] = {
    W0,
    W1, W1, W1, W1, W1, W1,
    W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2,
    W3, W3, W3, W3, W3, W3, W3, W3
};

// Moment powers for matrix method (each row is {px, py, pz})
static const int MomentsPower[QN][3] = {
    {0,0,0},{1,0,0},{2,0,0}, {0,1,0},{1,1,0},{2,1,0}, {0,2,0},{1,2,0},{2,2,0},
    {0,0,1},{1,0,1},{2,0,1}, {0,1,1},{1,1,1},{2,1,1}, {0,2,1},{1,2,1},{2,2,1},
    {0,0,2},{1,0,2},{2,0,2}, {0,1,2},{1,1,2},{2,1,2}, {0,2,2},{1,2,2},{2,2,2},
};

// ============================================================
//  Helper: integer power
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
//  Cell: stores distribution function + macroscopic quantities
// ============================================================
struct Cell {
    double f[QN];
    double rho;
    Vec3   vel;
    double T;
};

// ============================================================
//  Physical parameters
// ============================================================
struct PhysParams {
    double tau, dtau;
    double visc_atT;
    double dr, dt;
    int    stencilInterpWidth;
    int    stencilFixed;
    int    RegOrder;
    int    EquilibriumOrder;
    int    IsothermalRelaxation;
    int    fixedTemperature;
    int    MaxSteps;
    int    StepIterPeriod;

    // Shear wave initial condition parameters
    double u0, rho0, T0;
    double uDragX, uDragY, uDragZ;
    int    shearWaveDir;

    void setupUnits() {
        double ViscAtTUnitConv = 1.0 / dt;
        tau  = 0.5 + visc_atT * ViscAtTUnitConv;
        dtau = 1.0 / tau;
    }
};

// ============================================================
//  Global state
// ============================================================
static int Nx, Ny, Nz;
static std::vector<Cell> cells[2];   // double buffer [old, new]
static PhysParams PP;

inline int idx(int ix, int iy, int iz) {
    return ix + iy * Nx + iz * Nx * Ny;
}

// Periodic wrap
inline int wrap(int i, int N) { return ((i % N) + N) % N; }

// ============================================================
//  Equilibrium distribution f^eq
//  Hermite expansion up to EquilibriumOrder (1–4)
//  See data-inl.cu :: Cell::calcEq
// ============================================================
static void calcEq(double feq[QN], double Rho, Vec3 u, double Tempr) {
    if (Rho == 0) u = {0, 0, 0};

    const double dT  = dcs2;           // 1/cs2
    const double T0  = cs2;            // lattice temperature
    const double dT2 = dT * dT;
    const double dT4 = dT2 * dT2;
    const double u2  = dot(u, u);

    const int eno = PP.EquilibriumOrder;
    const int T1 = (eno >= 1), T2 = (eno >= 2), T3 = (eno >= 3), T4 = (eno >= 4);

    double Tcur = PP.IsothermalRelaxation ? cs2 : Tempr;
    const double mxwU = 1.0 - T2 * u2 * 0.5 * dT;

    for (int i = 0; i < QN; i++) {
        const Vec3 ei = ef[i];
        const double ei2 = dot(ei, ei);
        const double ei4 = ei2 * ei2;
        const double eu  = dot(ei, u);
        const double eu2 = eu * eu;
        const double eu4 = eu2 * eu2;
        const double dTc = Tcur - T0;

        double mxw = mxwU
            + T1 * eu * dT
            + T2 * eu2 * 0.5 * dT2
            + T2 * dTc * 0.5 * dT * (ei2 * dT - DIM)
            + T3 * (1.0/6.0) * eu * dT * (eu2*dT2 - 3*u2*dT + 3*dTc*dT*(ei2*dT - DIM - 2))
            + T4 * (1.0/24.0) * dT4 * (
                eu4 + 3*T0*T0*u2*u2 - 6*T0*eu2*u2
                + 6*dTc*eu2*ei2 + 3*dTc*dTc*ei4
                - 6*T0*dTc*dTc*(DIM+2)*ei2 + 3*T0*T0*dTc*dTc*DIM*(DIM+2)
                - 6*T0*dTc*u2*ei2 - 6*T0*dTc*(DIM+4)*eu2 + 6*T0*T0*dTc*(DIM+2)*u2
              );

        feq[i] = wt[i] * Rho * mxw;
    }
}

// ============================================================
//  BGK collision: f_new = f - (1/τ)(f − f^eq)
// ============================================================
static void collision(double f[QN], const double feq[QN]) {
    const double dtau = PP.dtau;
    for (int i = 0; i < QN; i++)
        f[i] = f[i] - dtau * (f[i] - feq[i]);
}

// ============================================================
//  Convergence check for inner iteration
// ============================================================
static bool isConv(const Cell& c1, const Cell& c2) {
    const double err_abs = 1e-12;
    const double err_rel = 1e-10;
    const double v1[] = {c1.rho, c1.vel.x, c1.vel.y, c1.vel.z, c1.T};
    const double v2[] = {c2.rho, c2.vel.x, c2.vel.y, c2.vel.z, c2.T};
    for (int i = 0; i < 5; i++)
        if (std::fabs(v1[i]-v2[i]) >= err_abs + err_rel*std::fabs(v1[i])) return false;
    for (int i = 0; i < QN; i++)
        if (std::fabs(c1.f[i]-c2.f[i]) >= err_abs + err_rel*std::fabs(c1.f[i])) return false;
    return true;
}

// ============================================================
//  Lagrange interpolation polynomial (tensor-product 3D)
//  Returns the weight for stencil point (ix,iy,iz) given
//  fractional shifts and stencil width N.
// ============================================================
static double LagrPol(int ix, int iy, int iz, Vec3 shifts, int N) {
    double a = 1.0;
    for (int p = 0; p < N; p++) if (p != ix) a *= (shifts.x - p) / (ix - p);
    for (int p = 0; p < N; p++) if (p != iy) a *= (shifts.y - p) / (iy - p);
    for (int p = 0; p < N; p++) if (p != iz) a *= (shifts.z - p) / (iz - p);
    return a;
}

// ============================================================
//  Hermite polynomial H_n(v; a,b,c,d,e)
//  Probabilists' Hermite tensors up to order 4
//  See momentsMatrix.cuh :: Hermite()
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
//  Hermite Tensor Coefficients
//  Stores coefficients a_n^{abc...} up to given order.
//  Total count for order N: 1 + 3 + 9 + 27 + ... = Σ 3^k
// ============================================================
static int tensor_count(int order) {
    int c = 0, d = 1;
    for (int n = 0; n <= order; n++) { c += d; d *= DIM; }
    return c;
}

struct TensorCoeffs {
    static const int MAX_COEFFS = 1 + 3 + 9 + 27 + 81 + 243; // up to order 5
    double k[MAX_COEFFS];
    int Order;

    TensorCoeffs() : Order(0) { std::memset(k, 0, sizeof(k)); }
    TensorCoeffs(int order, double val) : Order(order) {
        int nc = tensor_count(order);
        for (int i = 0; i < nc; i++) k[i] = val;
        for (int i = nc; i < MAX_COEFFS; i++) k[i] = 0;
    }

    int ncoeffs() const { return tensor_count(Order); }

    void operator+=(const TensorCoeffs& o) {
        int nc = ncoeffs();
        for (int i = 0; i < nc; i++) k[i] += o.k[i];
    }
    void operator*=(double s) {
        int nc = ncoeffs();
        for (int i = 0; i < nc; i++) k[i] *= s;
    }

    // Access coefficient for tensor of rank ni with indices abc[0..ni-1]
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

// Compute Hermite tensor coefficients from distribution function f
// gauge = (u, w) where v_iq = ef[iq]*w + u
static void calc_moments_tensors(const double gu[3], double gw,
                                 const double f[QN], TensorCoeffs& TC) {
    for (int ni = 0; ni <= TC.Order; ni++) {
        int Ndim = 1;
        for (int i = 0; i < ni; i++) Ndim *= DIM;

        for (int abc = 0; abc < Ndim; abc++) {
            int abci[5] = {0,0,0,0,0};
            for (int i = 0, Nds = 1; i < ni; i++, Nds *= DIM)
                abci[i] = (abc / Nds) % DIM;

            double& an = TC.get(ni, abci);
            an = 0;
            for (int iq = 0; iq < QN; iq++) {
                double v[3] = {ef[iq].x*gw + gu[0],
                               ef[iq].y*gw + gu[1],
                               ef[iq].z*gw + gu[2]};
                an += Hermite(ni, v, abci) * f[iq];
            }
        }
    }
}

// Convert A-coefficients (moving frame) → D-coefficients (reference frame)
// See momentsMatrix.cuh :: convertAtoD
static TensorCoeffs convertAtoD(TensorCoeffs& an, const double gu[3], double gw) {
    const double sqT  = gw;
    const double dsqT = 1.0 / sqT;
    const double T    = sqT * sqT;
    const double u[3] = {gu[0], gu[1], gu[2]};

    TensorCoeffs dn;
    dn.Order = an.Order;

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
                dn_val = an_val;
            } else if (ni == 1) {
                dn_val = dsqT * (an_val - u[a] * an0);
            } else if (ni == 2) {
                int ia[5] = {a,0,0,0,0}, ib[5] = {b,0,0,0,0};
                double An1_a = an.get(1, ia);
                double An1_b = an.get(1, ib);
                dn_val = dsqT * dsqT * (
                    an_val
                    - u[a] * An1_b - u[b] * An1_a
                    + (u[a]*u[b] - (T-1)*(a==b)) * an0
                );
            } else if (ni == 3) {
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

// Evaluate f_i from Hermite expansion: f_i = Σ (w_i / n!) d_n H_n(ef_i)
static double eval_fi_Hermit(TensorCoeffs& TC, int iq) {
    double fi = 0;
    const double ev[3] = {ef[iq].x, ef[iq].y, ef[iq].z};
    for (int ni = 0; ni <= TC.Order; ni++) {
        int Ndim = 1, nfact = 1;
        for (int i = 1; i <= ni; i++) { Ndim *= DIM; nfact *= i; }

        for (int abc = 0; abc < Ndim; abc++) {
            int abci[5] = {0,0,0,0,0};
            for (int i = 0, Nds = 1; i < ni; i++, Nds *= DIM)
                abci[i] = (abc / Nds) % DIM;

            double dn = TC.get(ni, abci);
            fi += wt[iq] / nfact * dn * Hermite(ni, ev, abci);
        }
    }
    return fi;
}

// ============================================================
//  Moments Matrix (for RegOrder < 0, full matrix inverse)
//  Augmented [M | I] → Gauss-Jordan → [I | M^{-1}]
// ============================================================
struct MomentsMatrix {
    double m[QN][QN * 2];

    // Build moment matrix: M_{ij} = v_j^{px_i} * v_j^{py_i} * v_j^{pz_i}
    void init(const double gu[3], double gw) {
        for (int i = 0; i < QN; i++)
            for (int j = 0; j < QN; j++) {
                double v[3] = {ef[j].x*gw + gu[0],
                               ef[j].y*gw + gu[1],
                               ef[j].z*gw + gu[2]};
                m[i][j] = ipow(v[0], MomentsPower[i][0])
                         * ipow(v[1], MomentsPower[i][1])
                         * ipow(v[2], MomentsPower[i][2]);
            }
    }

    // In-place Gauss-Jordan inversion
    void inverse() {
        for (int i = 0; i < QN; i++)
            for (int j = QN; j < 2*QN; j++)
                m[i][j] = (i == j - QN) ? 1.0 : 0.0;

        for (int i = 0; i < QN; i++) {
            // Partial pivoting
            int pivot = i;
            for (int j = i+1; j < QN; j++)
                if (std::fabs(m[j][i]) > std::fabs(m[pivot][i])) pivot = j;

            if (pivot != i)
                for (int k = 0; k < 2*QN; k++) std::swap(m[i][k], m[pivot][k]);

            // Eliminate
            for (int j = 0; j < QN; j++) {
                if (i != j) {
                    double ratio = m[j][i] / m[i][i];
                    for (int k = 0; k < 2*QN; k++) m[j][k] -= ratio * m[i][k];
                } else {
                    double div = 1.0 / m[i][i];
                    for (int k = 0; k < 2*QN; k++) m[i][k] *= div;
                }
            }
        }
    }

    // f_i = Σ_j M^{-1}_{i,j} * moments_j
    double get_inv(int irow, const double mvec[QN]) const {
        double fi = 0;
        for (int j = 0; j < QN; j++) fi += m[irow][j + QN] * mvec[j];
        return fi;
    }
};

// Compute moment vector: mom_i = Σ_j v_j^{p_i} * f_j
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
//  Initialization: Shear wave
//  See materials.cuh :: shear_wave()
// ============================================================
static void init_shear_wave() {
    cells[0].resize(Nx * Ny * Nz);
    cells[1].resize(Nx * Ny * Nz);

    for (int iz = 0; iz < Nz; iz++)
    for (int iy = 0; iy < Ny; iy++)
    for (int ix = 0; ix < Nx; ix++) {
        double vx = 0, vy = 0, vz = 0;

        if (PP.shearWaveDir == 1) {
            vx = PP.u0 * std::sin(2*M_PI * ix / Nx);
        } else if (PP.shearWaveDir == 2) {
            double u_l = PP.u0 * std::sin(2*M_PI * (ix+iy) / Nx);
            vx = -u_l / std::sqrt(2.0);
            vy =  u_l / std::sqrt(2.0);
        } else if (PP.shearWaveDir == 3) {
            double u_l = PP.u0 * std::sin(2*M_PI * (ix+iy+iz) / Nx);
            vx = -u_l / std::sqrt(6.0);
            vy = -u_l / std::sqrt(6.0);
            vz =  u_l * std::sqrt(2.0 / 3.0);
        }
        vx += PP.uDragX;
        vy += PP.uDragY;
        vz += PP.uDragZ;

        Cell& c   = cells[0][idx(ix, iy, iz)];
        c.rho = 1.0;
        c.vel = {vx, vy, vz};
        c.T   = PP.T0;

        double feq[QN];
        calcEq(feq, c.rho, c.vel, c.T);
        for (int iq = 0; iq < QN; iq++) c.f[iq] = feq[iq];

        cells[1][idx(ix, iy, iz)] = c;
    }
}

// ============================================================
//  POD Semi-Lagrangian Streaming + Collision (one time step)
//
//  Algorithm per cell (ix, iy, iz):
//    1. Inner iteration (fixed-point) to convergence:
//       a. Compute gauge = (vel, sqrt(T/TLat))
//       b. For each velocity direction iq:
//          - Back-trace: departure point = (ix,iy,iz) - v_iq
//          - Lagrange-interpolate in moment space (Hermite tensor
//            or full matrix) from surrounding cells
//          - Reconstruct new f[iq]
//       c. Update macroscopic rho, vel, T from new f
//       d. Check convergence
//    2. Collision: BGK relaxation toward f^eq(rho, 0, TLat)
//       (zero velocity because advection is already in streaming)
//    3. Store result in buffer[1]
// ============================================================
static void streaming_collision_step() {
    const int ild = 0;   // read buffer
    const int ist = 1;   // write buffer
    const int Npoints = PP.stencilInterpWidth + 1;

    for (int iz = 0; iz < Nz; iz++)
    for (int iy = 0; iy < Ny; iy++)
    for (int ix = 0; ix < Nx; ix++) {
        Cell cell = cells[ild][idx(ix, iy, iz)];
        Cell cell_new;

        // ---- Inner iteration for convergence ----
        int Niter = 0;
        while (Niter < 100) {
            const double gw = std::sqrt(cell.T / TLat);   // gauge scaling
            const double gu[3] = {cell.vel.x, cell.vel.y, cell.vel.z};

            // For RegOrder<0: build & invert moments matrix (once per iteration)
            MomentsMatrix Mm;
            if (PP.RegOrder < 0) {
                Mm.init(gu, gw);
                Mm.inverse();
            }

            // ---- Process each velocity direction ----
            for (int iq = 0; iq < QN; iq++) {
                // Actual velocity for direction iq in the lab frame
                Vec3 v = {ef[iq].x * gw + gu[0],
                          ef[iq].y * gw + gu[1],
                          ef[iq].z * gw + gu[2]};

                // Departure point (back-trace)
                Vec3 xf = {ix - v.x, iy - v.y, iz - v.z};

                // Determine interpolation stencil origin
                int sMinX, sMinY, sMinZ;
                if (PP.stencilFixed) {
                    sMinX = ix - PP.stencilInterpWidth / 2;
                    sMinY = iy - PP.stencilInterpWidth / 2;
                    sMinZ = iz - PP.stencilInterpWidth / 2;
                } else {
                    double hw = 0.5 * PP.stencilInterpWidth;
                    sMinX = (int)std::round(xf.x - hw);
                    sMinY = (int)std::round(xf.y - hw);
                    sMinZ = (int)std::round(xf.z - hw);
                }

                Vec3 shifts = {xf.x - sMinX, xf.y - sMinY, xf.z - sMinZ};

                if (PP.RegOrder < 0) {
                    // --- Full matrix inverse method ---
                    double val = 0;
                    for (int xs = 0; xs < Npoints; xs++)
                    for (int ys = 0; ys < Npoints; ys++)
                    for (int zs = 0; zs < Npoints; zs++) {
                        int cx = wrap(sMinX + xs, Nx);
                        int cy = wrap(sMinY + ys, Ny);
                        int cz = wrap(sMinZ + zs, Nz);
                        const Cell& sc = cells[ild][idx(cx, cy, cz)];

                        double coeff = LagrPol(xs, ys, zs, shifts, Npoints);

                        double ig[3] = {sc.vel.x, sc.vel.y, sc.vel.z};
                        double igw   = std::sqrt(sc.T / TLat);
                        double mVec[QN];
                        calc_moments_vec(ig, igw, sc.f, mVec);

                        val += coeff * Mm.get_inv(iq, mVec);
                    }
                    cell_new.f[iq] = val;
                } else {
                    // --- Hermite tensor regularization ---
                    TensorCoeffs an_interp(PP.RegOrder, 0.0);

                    for (int xs = 0; xs < Npoints; xs++)
                    for (int ys = 0; ys < Npoints; ys++)
                    for (int zs = 0; zs < Npoints; zs++) {
                        int cx = wrap(sMinX + xs, Nx);
                        int cy = wrap(sMinY + ys, Ny);
                        int cz = wrap(sMinZ + zs, Nz);
                        const Cell& sc = cells[ild][idx(cx, cy, cz)];

                        double coeff = LagrPol(xs, ys, zs, shifts, Npoints);

                        double ig[3] = {sc.vel.x, sc.vel.y, sc.vel.z};
                        double igw   = std::sqrt(sc.T / TLat);

                        TensorCoeffs an_p(PP.RegOrder, 0.0);
                        calc_moments_tensors(ig, igw, sc.f, an_p);
                        an_p *= coeff;
                        an_interp += an_p;
                    }

                    TensorCoeffs dn = convertAtoD(an_interp, gu, gw);
                    cell_new.f[iq] = eval_fi_Hermit(dn, iq);
                }
            }

            // ---- Compute macroscopic quantities from new f ----
            double sum_rho = 0, sum_vx = 0, sum_vy = 0, sum_vz = 0, sum_M2 = 0;
            for (int ik = 0; ik < QN; ik++) {
                Vec3 v_k = {ef[ik].x*gw + gu[0],
                            ef[ik].y*gw + gu[1],
                            ef[ik].z*gw + gu[2]};
                sum_rho += cell_new.f[ik];
                sum_vx  += v_k.x * cell_new.f[ik];
                sum_vy  += v_k.y * cell_new.f[ik];
                sum_vz  += v_k.z * cell_new.f[ik];
                sum_M2  += dot(v_k, v_k) * cell_new.f[ik];
            }
            cell_new.rho = sum_rho;
            cell_new.vel = {sum_vx/sum_rho, sum_vy/sum_rho, sum_vz/sum_rho};
            cell_new.T   = sum_M2/sum_rho - dot(cell_new.vel, cell_new.vel);
            cell_new.T  /= DIM;

            if (PP.fixedTemperature) cell_new.T = cell.T;
            if (cell_new.T < 0) {
                printf("Warning: negative T at (%d,%d,%d) iter %d, T=%g\n",
                       ix, iy, iz, Niter, cell_new.T);
                cell_new.T = -cell_new.T;
            }

            Niter++;
            if (isConv(cell, cell_new)) { cell = cell_new; break; }
            cell = cell_new;
        }

        // ---- Collision ----
        // In POD, equilibrium uses zero velocity and lattice temperature
        // because the semi-Lagrangian step already handles advection
        double feq[QN];
        calcEq(feq, cell.rho, {0, 0, 0}, TLat);
        collision(cell.f, feq);

        cells[ist][idx(ix, iy, iz)] = cell;
    }

    // Swap buffers
    std::swap(cells[0], cells[1]);
}

// ============================================================
//  Diagnostics: print conservation quantities
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
//  Main
// ============================================================
int main(int argc, char** argv) {
    // Defaults (matching shear_wave_test.py, but with smaller grid for CPU)
    Nx = 20;  Ny = 20;  Nz = 20;
    int MaxSteps = 100;
    int RegOrder = 2;

    if (argc > 1) Nx = std::atoi(argv[1]);
    if (argc > 2) Ny = std::atoi(argv[2]);
    if (argc > 3) Nz = std::atoi(argv[3]);
    if (argc > 4) MaxSteps = std::atoi(argv[4]);
    if (argc > 5) RegOrder = std::atoi(argv[5]);

    printf("=== POD Semi-Lagrangian LBM — Shear Wave Test (CPU) ===\n");
    printf("Grid: %d x %d x %d,  Steps: %d,  RegOrder: %d\n",
           Nx, Ny, Nz, MaxSteps, RegOrder);
    printf("Lattice: D3Q27 POD (TLat=%.1f, ec=sqrt(3)=%.6f)\n\n", TLat, EC);

    // Physical parameters (from shear_wave_test.py)
    PP.visc_atT            = 0.2;
    PP.dr                  = 1.0;
    PP.dt                  = 1.0;
    PP.stencilInterpWidth  = 2;
    PP.stencilFixed        = 0;
    PP.RegOrder            = RegOrder;
    PP.EquilibriumOrder    = 4;
    PP.IsothermalRelaxation= 1;
    PP.fixedTemperature    = 1;
    PP.MaxSteps            = MaxSteps;
    PP.StepIterPeriod      = 1;

    const double Tinit = 1.0 / 3.0;
    const double Ma_a  = 10.0;
    const int    waveD = 3;

    PP.T0           = Tinit;
    PP.u0           = 0.05;
    PP.rho0         = 1.0;
    PP.shearWaveDir = waveD;
    PP.uDragX = (Ma_a / std::sqrt((double)waveD)) * std::sqrt(Tinit);
    PP.uDragY = (waveD > 1) ? PP.uDragX : 0;
    PP.uDragZ = (waveD > 2) ? PP.uDragX : 0;
    PP.setupUnits();

    printf("  visc=%.4f  tau=%.6f  dtau=%.6f\n", PP.visc_atT, PP.tau, PP.dtau);
    printf("  T0=%.6f  u0=%.4f  Ma=%.1f  waveDir=%d\n", PP.T0, PP.u0, Ma_a, waveD);
    printf("  uDrag=(%.6f, %.6f, %.6f)\n\n", PP.uDragX, PP.uDragY, PP.uDragZ);

    // Initialize
    init_shear_wave();
    print_diagnostics(0);

    // Time-stepping loop
    for (int step = 1; step <= MaxSteps; step++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        streaming_collision_step();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (step % PP.StepIterPeriod == 0) {
            print_diagnostics(step);
            printf("  (%.1f ms, %.3f MLU/s)\n", ms, 1e-3*Nx*Ny*Nz / ms);
        }
    }

    printf("\nDone.\n");
    return 0;
}
