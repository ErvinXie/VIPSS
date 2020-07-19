#include "rbfcore.h"
#include "utility.h"
#include "Solver.h"
#include <armadillo>
#include <fstream>
#include <limits>
#include <iomanip>
#include <ctime>
#include <chrono>
#include<algorithm>
#include "ImplicitedSurfacing.h"
#include "OctotreeSurfacer.h"

typedef std::chrono::high_resolution_clock Clock;

void RBF_Core::BuildJ(RBF_Paras para) {

    isuse_sparse = para.isusesparse;
    sparse_para = para.sparse_para;
    Hermite_weight_smoothness = para.Hermite_weight_smoothness;
    Hermite_designcurve_weight = para.Hermite_designcurve_weight;
//    handcraft_sigma = para.handcraft_sigma;
//    wDir = para.wDir;
//    wOrt = para.wOrt;
//    wFlip = para.wFlip;
    curMethod = para.Method;

    Set_Actual_Hermite_LSCoef(para.Hermite_ls_weight);
    Set_Actual_User_LSCoef(para.user_lamnbda);
    isNewApprox = true;
    isnewformula = true;

    auto t1 = Clock::now();

    switch (curMethod) {

        case Hermite_UnitNormal:
            Set_Hermite_PredictNormal(pts);
            break;
    }
    auto t2 = Clock::now();
    cout << "Build Time: " << (setup_time = std::chrono::nanoseconds(t2 - t1).count() / 1e9) << endl << endl;
    if (0)BuildCoherentGraph();
}

void RBF_Core::AddPointsAndBuildJnew(vector<double> &newpts) {
    auto t1 = Clock::now();
    cout << "setting J" << endl;

    for (auto p:newpts) {
        pts.push_back(p);
    }

    arma::mat E, F, G;
    arma::mat B, D, A_hat_inv, C, newAinv;

    int k = newpts.size() / 3;
    int n = npt;
    double temp[9];

    {
        B.zeros(4 * n + 4, 4 * k);

        for (int i = 0; i < n; i++) {
            double *pi = pts.data() + i * 3;
            for (int j = 0; j < k; j++) {
                double *pj = newpts.data() + j * 3;
                B(i, j) = Kernal_Function_2p(pi, pj);

                Kernal_Gradient_Function_2p(pi, pj, temp);
                for (int l = 0; l < 3; l++) {
                    B(i, k + j * 3 + l) = temp[l];
                }

                Kernal_Gradient_Function_2p(pj, pi, temp);
                for (int l = 0; l < 3; l++) {
                    B(n + i * 3 + l, j) = temp[l];
                }

                Kernal_Hessian_Function_2p(pi, pj, temp);
                for (int p = 0; p < 3; p++) {
                    for (int q = 0; q < 3; q++) {
                        B(n + i * 3 + p, k + j * 3 + q) = -*(temp + p * 3 + q);
                    }
                }
            }
        }
        for (int i = 0; i < k; i++) {
            double *pi = pts.data() + i * 3;
            B(4 * n + 3, i) = 1;
            for (int j = 0; j < 3; j++) {
                B(4 * n + j, i) = pi[j];
                B(4 * n + j, i * 3 + j) = -1;
            }
        }
    }

    {
        cout << "k:" << k << endl;
        D.zeros(4 * k, 4 * k);
        for (int i = 0; i < k; i++) {
            double *pi = newpts.data() + i * 3;
            for (int j = 0; j < k; j++) {
                double *pj = newpts.data() + j * 3;

                D(i, j) = Kernal_Function_2p(pi, pj);
                Kernal_Gradient_Function_2p(pi, pj, temp);
                for (int l = 0; l < 3; l++) {
                    D(i, k + j * 3 + l) = temp[l];
                }

                Kernal_Gradient_Function_2p(pj, pi, temp);
                for (int l = 0; l < 3; l++) {
                    D(k + i * 3 + l, j) = temp[l];
                }

                Kernal_Hessian_Function_2p(pj, pi, temp);
                for (int p = 0; p < 3; p++) {
                    for (int q = 0; q < 3; q++) {
                        D(k + i * 3 + p, k + j * 3 + q) = -*(temp + p * 3 + q);
                    }
                }
            }
        }

        cout << D.n_cols << " " << D.n_rows << endl;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                cout << D(D.n_rows - 10 + i, D.n_rows - 10 + j) << "\t";
            }
            cout << endl;
        }

    }

    cout << det(D) << endl;

//    C = inv(D);
    {
        A_hat_inv.zeros(4 * (n + k) + 4, 4 * (n + k) + 4);
        arma::mat TEMP = inv(D - B.t() * Ainv * B);
        cout << det(TEMP) << endl;
        A_hat_inv.submat(4 * n + 4, 4 * n + 4, 4 * n + 4 + 4 * k - 1, 4 * n + 4 + 4 * k - 1)
                = TEMP;

        TEMP = (-Ainv) * B * TEMP;
        A_hat_inv.submat(0, 4 * n + 4, 4 * n + 4 - 1, 4 * n + 4 + 4 * k - 1)
                = TEMP;
        A_hat_inv.submat(4 * n + 4, 0, 4 * n + 4 + 4 * k - 1, 4 * n + 4 - 1)
                = TEMP.t();
        A_hat_inv.submat(0, 0, 4 * n + 4 - 1, 4 * n + 4 - 1)
                = Ainv - TEMP * B.t() * Ainv;
    }

    newAinv.zeros(4 * (n + k) + 4, 4 * (n + k) + 4);
    vector<vector<int> > submatrix_index = {
            {0,             0,             0,             0,             n,     n},//M00
            {0,             n,             0,             4 * n + 4,     n,     k},//M00
            {n,             n,             4 * n + 4,     4 * n + 4,     k,     k},//M00

            {0,             n + k,         0,             n,             n,     3 * n},//M01
            {0,             n + k + 3 * n, 0,             4 * n + 4 + k, n,     3 * k},//M01
            {n,             n + k,         4 * n + 4,     n,             k,     3 * n},//M01
            {n,             n + k + 3 * n, 4 * n + 4,     4 * n + 4 + k, k,     3 * k},//M01

            {0,             4 * (n + k),   0,             4 * n,         n,     4},//N0
            {n,             4 * (n + k),   4 * n + 4,     4 * n,         k,     4},//N0

            {n + k,         4 * (n + k),   n,             4 * n,         3 * n, 4},//N1
            {n + k + 3 * n, 4 * (n + k),   4 * n + 4 + k, 4 * n,         3 * k, 4},//N1


            {n + k,         n + k,         n,             n,             3 * n, 3 * n},//M11
            {n + k,         n + k + 3 * n, n,             4 * n + 4 + k, 3 * n, 3 * k},//M11
            {n + k + 3 * n, n + k + 3 * n, 4 * n + 4 + k, 4 * n + 4 + k, k,     k},//M11
    };
    for (const auto &v:submatrix_index) {
        newAinv.submat(v[0], v[1], v[0] + v[4] - 1, v[1] + v[5] - 1)
                = A_hat_inv.submat(v[2], v[3], v[2] + v[4] - 1, v[3] + v[5] - 1);
        newAinv.submat(v[1], v[0], v[1] + v[5] - 1, v[0] + v[4] - 1)
                = A_hat_inv.submat(v[3], v[2], v[3] + v[5] - 1, v[2] + v[4] - 1);
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << setprecision(3) << newAinv(i, j) << " ";
        }
        cout << endl;
    }
    Ainv = newAinv;
    npt = pts.size() / 3;

    Minv = Ainv.submat(0, 0, npt * 4 - 1, npt * 4 - 1);
    Ninv = Ainv.submat(0, npt * 4, (npt) * 4 - 1, (npt + 1) * 4 - 1);

    J = Minv;
    J00 = J.submat(0, 0, npt - 1, npt - 1);
    J01 = J.submat(0, npt, npt - 1, npt * 4 - 1);
    J11 = J.submat(npt, npt, npt * 4 - 1, npt * 4 - 1);

    M.clear();
    N.clear();
    cout << "J11: " << J11.n_cols << endl;


    //Set_Hermite_DesignedCurve();

    Set_User_Lamnda_ToMatrix(User_Lamnbda_inject);


//		arma::vec eigval, ny;
//		arma::mat eigvec;
//		ny = eig_sym( eigval, eigvec, J);
//		cout<<ny<<endl;
    cout << "J: " << J.n_cols << endl;

    cout << "Add solve J total: " << (setK_time = std::chrono::nanoseconds(Clock::now() - t1).count() / 1e9) << endl;
    return;
}

void RBF_Core::InitAndOptNormal(RBF_Paras para) {

    auto t1 = Clock::now();
    curInitMethod = para.InitMethod;
    cout << "Init Method: " << mp_RBF_INITMETHOD[curInitMethod] << endl;
    switch (curInitMethod) {
        case Lamnbda_Search:
            Lambda_Search_GlobalEigen();
            break;

    }
    auto t2 = Clock::now();
    cout << "Init Time: " << (init_time = std::chrono::nanoseconds(t2 - t1).count() / 1e9) << endl << endl;

    mp_RBF_InitNormal[curMethod == HandCraft ? 0 : 1][curInitMethod] = initnormals;

    OptNormal(0);
}

void RBF_Core::OptNormal(int method) {

    cout << "OptNormal" << endl;
    auto t1 = Clock::now();

    switch (curMethod) {
        case Hermite_UnitNormal:
            Opt_Hermite_PredictNormal_UnitNormal();
            break;

    }
    auto t2 = Clock::now();
    cout << "Opt Time: " << (solve_time = std::chrono::nanoseconds(t2 - t1).count() / 1e9) << endl << endl;
    if (method == 0)
        mp_RBF_OptNormal[curMethod == HandCraft ? 0 : 1][curInitMethod] = newnormals;
}


void RBF_Core::Surfacing(int method, int n_voxels_1d) {

    n_evacalls = 0;

    if (method == 0) {
        Surfacer sf;
        surf_time = sf.Surfacing_Implicit(pts, n_voxels_1d, false, RBF_Core::Dist_Function);
        sf.WriteSurface(finalMesh_v, finalMesh_fv);
    } else if (method == 1) {
        //todo: octree surfacing
        double re_time;
        cout << "Implicit Surfacing: " << endl;

        auto t1 = Clock::now();

        OctotreeSurfacer surfacer(pts, [this](tuple<double, double, double> p) -> double {
            return RBF_Core::Dist_Function(get<0>(p), get<1>(p), get<2>(p));
        }, n_voxels_1d);
        surfacer.Surfacing();
        finalMesh_v = surfacer.vertices;
        finalMesh_fv = surfacer.triangleFaces;
        cout << "Implicit Surfacing Done." << endl;
        auto t2 = Clock::now();
        cout << "Total Surfacing time: " << (re_time = std::chrono::nanoseconds(t2 - t1).count() / 1e9) << endl;
        surf_time = re_time;
//        debug_output << "Node count:" << surfacer.face_node[0].size() << endl;
    }
    cout << "n_evacalls: " << n_evacalls << "   ave: " << surf_time / n_evacalls << endl;
//    debug_output << "Surfacing time:" << surf_time << endl;
//    debug_output << "Implicit function calls:" << n_evacalls << endl;
//    debug_output << "Vertices:" << finalMesh_v.size() / 3 << endl;
//    debug_output << "Faces:" << finalMesh_fv.size() / 3 << endl;

}


int RBF_Core::InjectData(vector<double> &pts, RBF_Paras para) {

    vector<int> labels;
    vector<double> normals, tangents;
    vector<uint> edges;

    InjectData(pts, labels, normals, tangents, edges, para);

}

int RBF_Core::InjectData(vector<double> &pts, vector<int> &labels, vector<double> &normals, vector<double> &tangents,
                         vector<uint> &edges, RBF_Paras para) {

    isuse_sparse = para.isusesparse;
    sparse_para = para.sparse_para;
    //isuse_sparse = false;
    this->pts = pts;
    this->labels = labels;
    this->normals = normals;
    this->tangents = tangents;
    this->edges = edges;
    npt = this->pts.size() / 3;
    curMethod = para.Method;
    curInitMethod = para.InitMethod;

    polyDeg = para.polyDeg;
    User_Lambda = para.user_lamnbda;
    rangevalue = para.rangevalue;
    maxvalue = 10000;

    cout << "number of points: " << pts.size() / 3 << endl;
    cout << "normals: " << this->normals.size() << endl;
    sol.Statue = 1;
    Init(para.Kernal);

    SetSigma(para.sigma);

    SetThis();

    return 1;
}

int RBF_Core::ThreeStep(vector<double> &pts, vector<int> &labels, vector<double> &normals, vector<double> &tangents,
                        vector<uint> &edges, RBF_Paras para) {

    InjectData(pts, labels, normals, tangents, edges, para);
    BuildJ(para);
    InitAndOptNormal(para);
    OptNormal(0);

    return 1;
}


int RBF_Core::AllStep(vector<double> &pts, vector<int> &labels, vector<double> &normals, vector<double> &tangents,
                      vector<uint> &edges, RBF_Paras para) {

    InjectData(pts, labels, normals, tangents, edges, para);
    BuildJ(para);
    InitAndOptNormal(para);
    OptNormal(0);
    Surfacing(0, 100);
    return 1;

}

void RBF_Core::BatchInitEnergyTest(vector<double> &pts, vector<int> &labels, vector<double> &normals,
                                   vector<double> &tangents, vector<uint> &edges, RBF_Paras para) {

    InjectData(pts, labels, normals, tangents, edges, para);
    BuildJ(para);
    para.ClusterVisualMethod = 0;//RBF_Init_EMPTY
    for (int i = 0; i < RBF_Init_EMPTY; ++i) {
        para.InitMethod = RBF_InitMethod(i);
        InitAndOptNormal(para);
        OptNormal(0);
        Record();
    }
    Print_Record_Init();
}


vector<double> *RBF_Core::ExportPts() {

    return &pts;


}

vector<double> *RBF_Core::ExportPtsNormal(int normal_type) {

    if (normal_type == 0)return &normals;
    else if (normal_type == 1)return &initnormals;
    else if (normal_type == 2)return &initnormals_uninorm;
    else if (normal_type == 3)return &newnormals;

    return NULL;
}


vector<double> *RBF_Core::ExportInitNormal(int kmethod, RBF_InitMethod init_type) {

    if (mp_RBF_InitNormal[kmethod].find(init_type) != mp_RBF_InitNormal[kmethod].end())
        return &(mp_RBF_InitNormal[kmethod][init_type]);
    else return NULL;
}

vector<double> *RBF_Core::ExportOptNormal(int kmethod, RBF_InitMethod init_type) {

    if (mp_RBF_OptNormal[kmethod].find(init_type) != mp_RBF_OptNormal[kmethod].end())
        return &(mp_RBF_OptNormal[kmethod][init_type]);
    else return NULL;
}


void RBF_Core::Print_Record_Init() {

    cout << "InitMethod" << string(30 - string("InitMethod").size(), ' ') << "InitEn\t\t FinalEn" << endl;
    cout << std::setprecision(8);
    {
        for (int i = 0; i < record_initmethod.size(); ++i) {
            cout << record_initmethod[i] << string(30 - record_initmethod[i].size(), ' ') << record_initenergy[i]
                 << "\t\t" << record_energy[i] << endl;
        }
    }

}
