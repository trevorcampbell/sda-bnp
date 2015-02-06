#ifndef __MATCHING_HPP
#define __MATCHING_HPP
#include <stdint.h>
#include <vector>

typedef MXi Eigen::MatrixXi;
class HungarianMatching{
	public:
		HungarianMatching(MXi costs);
		int run(std::vector<int>& matchings);
	private:
		MXi costs;
};

#endif
