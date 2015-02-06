#ifndef __MATCHING_HPP
#define __MATCHING_HPP
#include <stdint.h>
#include <vector>

typedef MXi Eigen::MatrixXi;
int hungarian(MXi costs, std::vector<int>& matchings);

#endif
