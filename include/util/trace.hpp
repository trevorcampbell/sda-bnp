#ifndef __TRACE_HPP
#include<string>
#include<fstream>
#include<sstream>
#include<iostream>

class Trace{
	public:
		std::vector<double> times, objs, testlls;
		void clear(){
			times.clear();
			objs.clear();
			testlls.clear();
		}
		void save(std::string name){
			std::ofstream out_trc(name+"-trace.log", std::ios_base::trunc);
			for (uint32_t i = 0; i < times.size(); i++){
				out_trc << times[i] << " " << objs[i];
				if (i < testlls.size()){
					out_trc << " " << testlls[i] << std::endl;
				} else {
					out_trc << std::endl;
				}
			}
			out_trc.close();
		}
};

class MultiTrace{
	public:
		std::vector<double> globaltimes, globaltestlls;
		std::vector<uint32_t> globalclusters, globalmatchings;
		std::vector< std::vector<double> > localtimes, localobjs, localtestlls;
		std::vector<double> localstarttimes;
		void clear(){
			globaltimes.clear();
			globaltestlls.clear();
			globalclusters.clear();
			globalmatchings.clear();
			localtimes.clear();
			localobjs.clear();
			localtestlls.clear();
			localstarttimes.clear();
		}
		void save(std::string name){
			//for a multitrace, the only thing to output is testll
			//so if this isn't the right size, forget it
			if (globaltimes.size() != globaltestlls.size()){
				std::cout << "Error -- globaltimes.size() != globaltestlls.size()!" << std::endl;
				return;
			}
			std::ofstream out_trc(name+"-globaltrace.log", std::ios_base::trunc);
			for (uint32_t i = 0; i < globaltimes.size(); i++){
				out_trc << globaltimes[i] << " " << globaltestlls[i] << " " << globalclusters[i] << " " << globalmatchings[i] << std::endl;
			}
			out_trc.close();

			std::ofstream out_tms(name + "-localstarttimes.log", std::ios_base::trunc);
			for (uint32_t i = 0; i < localstarttimes.size(); i++){
				out_tms << localstarttimes[i] << std::endl;
			}
			out_tms.close();

			for (uint32_t i = 0; i < localtimes.size(); i++){
				std::ostringstream oss;
				oss << name << "-localtrace-" << i << ".log";
				std::ofstream out_ltrc(oss.str().c_str(), std::ios_base::trunc);
				for (uint32_t j = 0; j < localtimes[i].size(); j++){
					out_ltrc << localtimes[i][j] << " " << localobjs[i][j];
					if (j < localtestlls[i].size()){
						out_ltrc << " " << localtestlls[i][j] << std::endl;
					} else {
						out_ltrc << std::endl;
					}
				}
				out_ltrc.close();
			}
		}
};
#define __TRACE_HPP
#endif /* __TRACE_HPP */
