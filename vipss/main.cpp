#include <iostream>
#include <unistd.h>
#include "src/rbfcore.h"
#include "src/readers.h"

using namespace std;


void SplitPath(const std::string &fullfilename, std::string &filepath);

void SplitFileName(const std::string &fullfilename, std::string &filepath, std::string &filename, std::string &extname);

RBF_Paras Set_RBF_PARA();

//ofstream debug_output;

int main(int argc, char **argv) {
    cout << argc << endl;
//    debug_output.open("./debug_out.txt", ios::app);

    string infilename;
    string outpath, pcname, ext, inpath;

    int n_voxel_line = 100;

    double user_lambda = 0;

    bool is_surfacing = false;
    bool is_outputtime = false;
    int surfacing_method = 1;

    int c;
    optind = 1;
    while ((c = getopt(argc, argv, "i:o:l:s:tm:")) != -1) {
        switch (c) {
            case 'i':
                infilename = optarg;
                break;
            case 'o':
                outpath = string(optarg);
                break;
            case 'l':
                user_lambda = atof(optarg);
                printf("%s\n", optarg);
                break;
            case 's':
                is_surfacing = true;
                n_voxel_line = atoi(optarg);
                printf("%s\n", optarg);
                break;
            case 't':
                is_outputtime = true;
                break;
            case 'm':
                surfacing_method = atoi(optarg);
                break;
            case '?':
                cout << "Bad argument setting!" << endl;
                break;
        }
    }

    if (outpath.empty())SplitFileName(infilename, outpath, pcname, ext);
    else SplitFileName(infilename, inpath, pcname, ext);
    cout << "input file: " << infilename << endl;
    cout << "output path: " << outpath << endl;

    cout << "user lambda: " << user_lambda << endl;
    cout << "is surfacing: " << is_surfacing << endl;
    cout << "surfacing method: " << map<int, string>({{0, "Bloomenthal"},
                                                      {1, "Adaptive Octree"}})[surfacing_method] << endl;

    cout << "number of voxel per D: " << n_voxel_line << endl;


    vector<double> Vs;
    RBF_Core rbf_core;
    RBF_Paras para = Set_RBF_PARA();
    para.user_lamnbda = user_lambda;

    readXYZ(infilename, Vs);

    vector<vector<double> > vss;
    vss.resize(2);
    for (int i = 0; i < Vs.size(); i++) {
//        if (i < (Vs.size() / 3 / 4) * 3) {
//            vss[0].push_back(Vs[i]);
//        }else{
//            vss[1].push_back(Vs[i]);
//        }
        if ((i / 3) % 4 == 0) {
            vss[0].push_back(Vs[i]);
        } else {
            vss[1].push_back(Vs[i]);
        }
    }
    cout << "vss sizes: " << Vs.size() << " " << vss[0].size() << " " << vss[1].size() << endl;
    cout << Vs.front() << endl;
    rbf_core.InjectData(Vs, para);

    rbf_core.BuildJ(para);
//    rbf_core.AddPointsAndBuildJnew(vss[1]);

    rbf_core.InitAndOptNormal(para);

    rbf_core.Write_Hermite_NormalPrediction(outpath + pcname + "_normal", 1);

//
//    debug_output << endl;
//    debug_output << infilename << endl;
//    debug_output << "method: " << surfacing_method << endl;
//    debug_output << "resolution: " << n_voxel_line << endl;
    if (is_surfacing) {
        rbf_core.Surfacing(surfacing_method, n_voxel_line);
        rbf_core.Write_Surface(outpath + pcname + "_surface");
    }


    if (is_outputtime) {
        rbf_core.Print_TimerRecord_Single(outpath + pcname + "_time.txt");
    }


    return 0;
}


RBF_Paras Set_RBF_PARA() {

    RBF_Paras para;
    RBF_InitMethod initmethod = Lamnbda_Search;

    RBF_Kernal Kernal = XCube;
    int polyDeg = 1;
    double sigma = 0.9;
    double rangevalue = 0.001;

    para.Kernal = Kernal;
    para.polyDeg = polyDeg;
    para.sigma = sigma;
    para.rangevalue = rangevalue;
    para.Hermite_weight_smoothness = 0.0;
    para.Hermite_ls_weight = 0;
    para.Hermite_designcurve_weight = 00.0;
    para.Method = RBF_METHOD::Hermite_UnitNormal;


    para.InitMethod = initmethod;

    para.user_lamnbda = 0;

    para.isusesparse = false;


    return para;
}


inline void
SplitFileName(const std::string &fullfilename, std::string &filepath, std::string &filename, std::string &extname) {
    int pos;
    pos = fullfilename.find_last_of('.');
    filepath = fullfilename.substr(0, pos);
    extname = fullfilename.substr(pos);
    pos = filepath.find_last_of("\\/");
    filename = filepath.substr(pos + 1);
    pos = fullfilename.find_last_of("\\/");
    filepath = fullfilename.substr(0, pos + 1);
    //cout<<modname<<' '<<extname<<' '<<filepath<<endl;
}

void SplitPath(const std::string &fullfilename, std::string &filepath) {

    std::string filename;
    std::string extname;
    SplitFileName(fullfilename, filepath, filename, extname);
}
