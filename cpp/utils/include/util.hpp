#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <list>
namespace util {

class Timer {
   private:
    clock_t tstart, tend;

   public:
    void start();
    void end();
};

// std::string path = util::current_path();
std::string current_path();

// std::string out = util::pathjoin({path, "b", "c"});
std::string pathjoin(const std::list<std::string>& strs);

bool isfile(const std::string& name);

}  // namespace util

#endif