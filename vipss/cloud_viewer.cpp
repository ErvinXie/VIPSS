#include <iostream>
#include <fstream>

using namespace std;

int main() {
    ifstream in("PointCloud.xyz");
    ofstream out("PC.xyz");
    double p[3];
    int cnt = 0;
    while (in >> p[cnt]) {
        cnt++;
        if (cnt == 3) {
            cnt = 0;
            if (p[0] < -0.2 && p[2] < -0.1) {
                out << p[0] << " " << p[1] << " " << p[2] << endl;
            }
        }
    }
    return 0;
}