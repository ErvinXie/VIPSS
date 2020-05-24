//
// Created by 谢威宇 on 2020/5/12.
//

#ifndef VIPSS_OCTOTREESURFACER_H
#define VIPSS_OCTOTREESURFACER_H

#include <vector>
#include <functional>
#include <map>
#include <fstream>
#include <string>

using namespace std;

class OctotreeSurfacer;


class OctNode;

typedef OctNode *np;

class OctNode {
public:
    OctNode(OctotreeSurfacer *surfacer, np father, const vector<int> &range);

    OctotreeSurfacer *surfacer;
    vector<int> range;
    vector<vector<int> > ps;

    bool containInput;
    bool end;
    bool split;
    bool visited = false;
    vector<int> corner_signs;
    vector<int> face_signs;

    vector<int> get_signs(bool prevent);

    void update_surround_signs();

    bool intersect();

    void update_sign(np sonp);


    np father;

    vector<np> sons;


    int get_son_idx(double ix, double iy, double iz);

    np &get_son(double ix, double iy, double iz);

    void addInputPoint(double ix, double iy, double iz);


    void Surfacing(np caller);

    tuple<int, int, int, int, int> get_face(int dir);

    void debug(int d = 0);

};


class OctotreeSurfacer {
public:
    const vector<double> &inputPoints;
    vector<double> upper, lower, step;
    ofstream out;

    int n_voxels_1d;

    function<double(tuple<double, double, double>)> implicitFunction;
    vector<double> vertices;
    vector<int> triangleFaces;
    vector<np> cubes;

    np octroot;


    map<tuple<int, int, int, int, int>, np> face_node[6];

    np get_np_by_face(np c, int dir);

    vector<np> get_neighbors(np c);

    OctotreeSurfacer(const vector<double> &inputPoints,
                     const function<double(tuple<double, double, double>)> &implicitFunction, int n_voxels_1d,
                     string out_path = "PointCloud.xyz");


    tuple<double, double, double> idx_to_coord(double x, double y, double z);

    tuple<double, double, double> idx_to_coord(tuple<double, double, double> idx);


    map<tuple<int, int, int>, double> mem;

    double get_function_value(tuple<int, int, int>);


    void Surfacing();

};


#endif //VIPSS_OCTOTREESURFACER_H
