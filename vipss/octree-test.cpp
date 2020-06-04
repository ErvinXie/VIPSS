//
// Created by 谢威宇 on 2020/5/14.
//
#include "OctotreeSurfacer.h"
#include <cmath>

int main() {
    OctotreeSurfacer S({1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1},
                       [](tuple<double, double, double> p) -> double {
                           auto &x = get<0>(p), &y = get<1>(p), &z = get<2>(p);
                           return min(sqrt((x - 0.33) * (x - 0.33) + y * y + z * z) - 0.25,
                                      sqrt((x + 0.33) * (x + 0.33) + y * y + z * z) - 0.25);
                       }, 50);
//    OctotreeSurfacer S({1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1},
//                       [](tuple<double, double, double> p) -> double {
//                           auto &x = get<0>(p), &y = get<1>(p), &z = get<2>(p);
//                           return sqrt(x * x + 9 * y * y + 9 * z * z) - 0.81;
//                       }, 100);
    S.Surfacing();
    return 0;
}