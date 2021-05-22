#include "util.hpp"
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <list>
#include <vector>

namespace util {

void Timer::start() {
    tstart = clock();
}
void Timer::end() {
    tend = clock();
    double dur;
    dur = (double)(tend - tstart);
    std::cout << "Cost Time:" << (dur / CLOCKS_PER_SEC) << "\n";
}

std::string current_path() {
    char* buffer;
    buffer = getcwd(NULL, 0);
    return buffer;
}

std::string pathjoin(const std::list<std::string>& strs) {
    std::string res = "";
    int cnt = 0;
    for (std::string s : strs) {
        if (cnt == 0) {
            res += s;
        } else {
            if (s[0] != '/') {
                res += ("/" + s);
            } else {
                res += s;
            }
        }
        cnt++;
    }
    return res;
}

bool isfile(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}
}  // namespace util