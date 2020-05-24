//
// Created by 谢威宇 on 2020/5/12.
//

#include "OctotreeSurfacer.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>

const double EPS = 1e-6;

bool equal(double a, double b) {
    return fabs(a - b) < EPS;
}

bool ge(double a, double b) {
    return equal(a, b) || a > b;
}

bool le(double a, double b) {
    return equal(a, b) || a < b;
}

OctotreeSurfacer::OctotreeSurfacer(const vector<double> &inputPoints,
                                   const function<double(tuple<double, double, double>)> &implicitFunction,
                                   int n_voxels_1d,
                                   string out_path)
        : inputPoints(inputPoints),
          implicitFunction(implicitFunction),
          n_voxels_1d(n_voxels_1d) {


    out = ofstream(out_path);
    for (int i = 0; i < 3; i++) {
        upper.push_back(numeric_limits<double>::min());
        lower.push_back(numeric_limits<double>::max());
    }
    for (int i = 0; i < inputPoints.size() / 3; i++) {
        for (int j = 0; j < 3; j++) {
            auto coord = inputPoints[i * 3 + j];
            upper[j] = max(upper[j], coord);
            lower[j] = min(lower[j], coord);
        }
    }
    for (int i = 0; i < 3; i++) {
        upper[i] += 0.05;
        lower[i] -= 0.05;
        step.push_back((upper[i] - lower[i]) / (double) n_voxels_1d);
//        cout<<upper[i]<<" "<<lower[i]<<" "<<step[i]<<endl;
    }

    octroot = new OctNode(this, nullptr, {0, n_voxels_1d, 0, n_voxels_1d, 0, n_voxels_1d});
    for (int i = 0; i < inputPoints.size() / 3; i++) {
        octroot->addInputPoint((inputPoints[i * 3] - lower[0]) / step[0],
                               (inputPoints[i * 3 + 1] - lower[1]) / step[1],
                               (inputPoints[i * 3 + 2] - lower[2]) / step[2]);
    }
//    octroot->debug();
}

double OctotreeSurfacer::get_function_value(tuple<int, int, int> idx) {
    if (!mem.count(idx)) {
        mem[idx] = implicitFunction(idx_to_coord(idx));
    }
    return mem[idx];
}

void OctotreeSurfacer::Surfacing() {
    mem.clear();
    octroot->Surfacing(nullptr);


}

tuple<double, double, double> OctotreeSurfacer::idx_to_coord(double x, double y, double z) {
    return {lower[0] + x * step[0], lower[1] + y * step[1], lower[2] + z * step[2]};
}

tuple<double, double, double> OctotreeSurfacer::idx_to_coord(tuple<double, double, double> idx) {
    return idx_to_coord(get<0>(idx), get<1>(idx), get<2>(idx));
}


np OctotreeSurfacer::get_np_by_face(np c, int dir) {
    auto face = c->get_face(dir);
    if (face_node[dir ^ 1].count(face) == 0)
        return nullptr;
    else
        return face_node[dir ^ 1][face];
}

vector<np> OctotreeSurfacer::get_neighbors(np c) {
    vector<np> re;
    for (int i = 0; i < 6; i++) {
        re.push_back(get_np_by_face(c, i));
    }
    return re;
}


OctNode::OctNode(OctotreeSurfacer *surfacer, np father, const vector<int> &range) :
        surfacer(surfacer),
        father(father),
        range(range) {
    containInput = false;
    visited = false;
    split = false;

    ps = {{range[0], range[1]},
          {range[2], range[3]},
          {range[4], range[5]}};

    end = true;
    for (auto &p:ps) {
        auto dp = p[1] - p[0];
        if (dp > 1) {
//            if (dp % 2 == 1) {
//                p.insert(p.begin() + 1, p[0] + dp / 2 + 1);
//            }
            p.insert(p.begin() + 1, p[0] + dp / 2);
            end = false;
        }
    }

    if (!end) {
        for (int i = 0; i < ps[0].size() - 1; i++) {
            for (int j = 0; j < ps[1].size() - 1; j++) {
                for (int k = 0; k < ps[2].size() - 1; k++) {
                    sons.push_back(nullptr);
                }
            }
        }
    }
    for (int i = 0; i < 6; i++) {
        surfacer->face_node[i][get_face(i)] = this;
    }
    for (int i = 0; i < 8; i++) {
        corner_signs.push_back(0);
    }
    int d = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto v = surfacer->get_function_value({ps[0][i], ps[1][j], ps[2][k]});
                if (equal(v, 0)) {
                    corner_signs[d] |= 3;
                } else if (v < 0) {
                    corner_signs[d] |= 2;
                } else if (v > 0) {
                    corner_signs[d] |= 1;
                }
                d++;
            }
        }
    }

    for (int i = 0; i < 6; i++)
        face_signs.push_back(0);

    int idx[6][4] = {{4, 5, 6, 7},
                     {0, 1, 2, 3},
                     {2, 3, 6, 7},
                     {0, 1, 4, 5},
                     {1, 3, 5, 7},
                     {0, 2, 4, 6}};
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 4; j++)
            face_signs[i] |= corner_signs[idx[i][j]];


}

int OctNode::get_son_idx(double ix, double iy, double iz) {
    int d = 0;
    for (int i = 0; i < ps[0].size() - 1; i++) {
        for (int j = 0; j < ps[1].size() - 1; j++) {
            for (int k = 0; k < ps[2].size() - 1; k++) {
                if (ix >= i && iy >= j && iz >= k) {
                    return d;
                }
                d++;
            }
        }
    }
    return -1;
}

np &OctNode::get_son(double ix, double iy, double iz) {
    return sons[get_son_idx(ix, iy, iz)];
}

void OctNode::addInputPoint(double ix, double iy, double iz) {
    containInput = true;
    if (!end) {
        int d = 0;
        for (int i = 0; i < ps[0].size() - 1; i++) {
            for (int j = 0; j < ps[1].size() - 1; j++) {
                for (int k = 0; k < ps[2].size() - 1; k++) {
                    if (ge(ix, ps[0][i]) && ge(iy, ps[1][j]) && ge(iz, ps[2][k])
                        && le(ix, ps[0][i + 1]) && le(iy, ps[1][j + 1]) && le(iz, ps[2][k + 1])) {
                        if (sons[d] == nullptr) {
                            sons[d] = new OctNode(surfacer, this, {ps[0][i], ps[0][i + 1],
                                                                   ps[1][j], ps[1][j + 1],
                                                                   ps[2][k], ps[2][k + 1]});
                        }
                        sons[d]->addInputPoint(ix, iy, iz);
                    }
                    d++;
                }
            }
        }
    }
}

void OctNode::debug(int d) {
    string indent;
    for (int i = 0; i < d; i++)
        indent += "  ";
    cout << indent;
    for (auto b:range) {
        cout << b << " ";
    }
    cout << endl;
    for (auto p:sons) {
        if (p != nullptr) {
            p->debug(d + 1);
        }
    }
}


void OctNode::Surfacing(np caller) {
    if (split)
        return;

    visited = true;
    bool intersect_with_surface = intersect();

//    for (int i = 0; i < 6; i++) {
//        cout << range[i] << " ";
//        if (i & 1) {
//            cout << "|\t";
//        }
//    }
//    cout << "input: " << containInput << " intersect: " << intersect_with_surface << endl;

    if (containInput || intersect_with_surface) {
        if (!end) {
            split = true;

            int d = 0;
            for (int i = 0; i < ps[0].size() - 1; i++) {
                for (int j = 0; j < ps[1].size() - 1; j++) {
                    for (int k = 0; k < ps[2].size() - 1; k++) {
                        if (sons[d] == nullptr) {
                            sons[d] = new OctNode(
                                    surfacer, this,
                                    {ps[0][i], ps[0][i + 1], ps[1][j], ps[1][j + 1], ps[2][k], ps[2][k + 1]});
                        }
                        d++;
                    }
                }
            }

//            bool splited = true;
//            while (splited) {
//                splited = false;
//                for (auto s:sons) {
//                    if (!s->split) {
//                        s->Surfacing(this);
//                        splited = s->split;
//                    }
//                }
//            }
            for (auto s:sons) {
                s->Surfacing(this);
            }


            d = 0;
            for (int i = 0; i < ps[0].size() - 1; i++) {
                for (int j = 0; j < ps[1].size() - 1; j++) {
                    for (int k = 0; k < ps[2].size() - 1; k++) {
                        if (i == 0) {
                            face_signs[1] |= sons[d]->get_signs(false)[1];
                        }
                        if (i == ps[0].size() - 2) {
                            face_signs[0] |= sons[d]->get_signs(false)[0];
                        }
                        if (j == 0) {
                            face_signs[3] |= sons[d]->get_signs(false)[3];
                        }
                        if (j == ps[1].size() - 2) {
                            face_signs[2] |= sons[d]->get_signs(false)[2];
                        }
                        if (k == 0) {
                            face_signs[5] |= sons[d]->get_signs(false)[5];
                        }
                        if (k == ps[2].size() - 2) {
                            face_signs[4] |= sons[d]->get_signs(false)[4];
                        }
                        d++;
                    }
                }
            }

            if (caller != father) {
                father->update_sign(this);
            }
//            auto neighbors = surfacer->get_neighbors(this);
//            for (int i = 0; i < 6; i++) {
//                if (neighbors[i] != nullptr) {
//                    neighbors[i]->get_signs(false);
//                    neighbors[i]->signs[i ^ 1] |= signs[i];
//                }
//            }

            for (auto n:surfacer->get_neighbors(this)) {
                if (n != nullptr && n->visited && n->split == false) {
//                    for (int i = 0; i < 6; i++) {
//                        cout << range[i] << " ";
//                        if (i & 1) {
//                            cout << "|\t";
//                        }
//                    }
//                    cout << end << " jump " << n->end << endl;
                    n->Surfacing(this);
                }
            }


        } else {
            if (intersect_with_surface == true) {
//                cout << "*" << endl;
//                surfacer->cubes.push_back(this);
                for (int i = 0; i < 3; i++) {
                    surfacer->out << surfacer->lower[i] + (ps[i][0] + ps[i][1]) / 2. * surfacer->step[i] << " ";
                }
                surfacer->out << endl;
            }
        }
    }
//    else {
//        if (containInput == true) {
//            for (auto x:range) {
//                cout << x << " ";
//            }
//            cout << endl;
//        }
//    }
}

tuple<int, int, int, int, int> OctNode::get_face(int dir) {
    if (dir == 0) {
        return {range[1], range[2], range[3], range[4], range[5]};
    } else if (dir == 1) {
        return {range[0], range[2], range[3], range[4], range[5]};
    } else if (dir == 2) {
        return {range[3], range[0], range[1], range[4], range[5]};
    } else if (dir == 3) {
        return {range[2], range[0], range[1], range[4], range[5]};
    } else if (dir == 4) {
        return {range[5], range[0], range[1], range[2], range[3]};
    } else if (dir == 5) {
        return {range[4], range[0], range[1], range[2], range[3]};
    }
}

template<typename first_type, typename tuple_type, size_t ...index>
auto to_vector_helper(const tuple_type &t, std::index_sequence<index...>) {
    return std::vector<first_type>{
            std::get<index>(t)...
    };
}

template<typename first_type, typename ...others>
auto to_vector(const std::tuple<first_type, others...> &t) {
    typedef typename std::remove_reference<decltype(t)>::type tuple_type;

    constexpr auto s = std::tuple_size<tuple_type>::value;

    return to_vector_helper<first_type, tuple_type>(t, std::make_index_sequence<s>{});
}

vector<int> OctNode::get_signs(bool prevent) {
    if (prevent == false) {
        for (int i = 0; i < 6; i++) {
            auto neighbor = surfacer->get_np_by_face(this, i);
            if (neighbor != nullptr && neighbor->split) {
                face_signs[i] |= neighbor->get_signs(true)[i ^ 1];
            }
        }
    }
    return face_signs;
}

bool OctNode::intersect() {
    int re = 0;
    for (auto s:get_signs(false)) {
        re |= s;
    }
    return re == 3;
}


void OctNode::update_sign(np sonp) {
    int d = 0;
    for (int i = 0; i < ps[0].size() - 1; i++) {
        for (int j = 0; j < ps[1].size() - 1; j++) {
            for (int k = 0; k < ps[2].size() - 1; k++) {
                if (sons[d] == sonp) {
                    if (i == 0) {
                        face_signs[1] |= sons[d]->get_signs(false)[1];
                    }
                    if (i == ps[0].size() - 2) {
                        face_signs[0] |= sons[d]->get_signs(false)[0];
                    }
                    if (j == 0) {
                        face_signs[3] |= sons[d]->get_signs(false)[3];
                    }
                    if (j == ps[1].size() - 2) {
                        face_signs[2] |= sons[d]->get_signs(false)[2];
                    }
                    if (k == 0) {
                        face_signs[5] |= sons[d]->get_signs(false)[5];
                    }
                    if (k == ps[2].size() - 2) {
                        face_signs[4] |= sons[d]->get_signs(false)[4];
                    }
                    if (father != nullptr) {
                        father->update_sign(this);
                    }
                    return;
                }
                d++;
            }
        }
    }
}

void OctNode::update_surround_signs() {
    auto neighbors = surfacer->get_neighbors(this);
    for (int i = 0; i < 6; i++) {

    }
}




