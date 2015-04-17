#ifndef __SDABNP_HPP
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <random>
#include <sdabnp/util/timer.hpp>
#include <sdabnp/util/kmpp.hpp>
#include <sdabnp/util/trace.hpp>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
double boost_lbeta(double a, double b);
using boost::math::digamma;
using boost::math::lgamma;



template<class Distribution, class Job>
class SDABNP{

};

#define __SDABNP_HPP
#endif /* __SDABNP_HPP */
