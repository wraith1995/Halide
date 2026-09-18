#include "Halide.h"
#include "Solve.h"        // Internal solve_for_outer/inner_interval
#include <iostream>
using namespace Halide;
using namespace Halide::Internal;
static void show(const char*l, Interval i){ std::cout<<l<<": ["<<i.min<<", "<<i.max<<"]\n"; }
int main(){
    Expr bv = Variable::make(Int(32), "bv");
    Expr m = min(bv < 0, bv < 2);              // AND-of-bools as min (what and_condition_over_domain emits)
    std::cout << "min(bv<0,bv<2) simplified: " << simplify(m) << "\n";
    std::cout << "!min(...)      simplified: " << simplify(!m) << "\n";
    show("solve_outer(!min(bv<0,bv<2))", solve_for_outer_interval(!m, "bv"));
    show("solve_outer(bv>=0)          ", solve_for_outer_interval(bv >= 0, "bv"));
    show("solve_outer(!(bv<0))        ", solve_for_outer_interval(!(bv < 0), "bv"));
    return 0;
}
