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
			if (testlls.size() == times.size()){
				for (uint32_t i = 0; i < times.size(); i++){
					out_trc << times[i] << " " << objs[i] << " " << testlls[i] << std::endl;
				}
			} else {
				for (uint32_t i = 0; i < times.size(); i++){
					out_trc << times[i] << " " << objs[i] << std::endl;
				}
			}
			out_trc.close();
		}
};

class MultiTrace{
	public:
		std::vector<double> times, testlls, mergetimes, starttimes;
		std::vector<uint32_t> clusters, matchings;
		void clear(){
			times.clear();
			testlls.clear();
			clusters.clear();
			matchings.clear();
			mergetimes.clear();
			starttimes.clear();
		}
		void save(std::string name){
			//for a multitrace, the only thing to output is testll
			//so if this isn't the right size, forget it
			if (times.size() != testlls.size()){
				std::cout << "Error -- times.size() != testlls.size()!" << std::endl;
				return;
			}
			std::ofstream out_trc(name+"-trace.log", std::ios_base::trunc);
			for (uint32_t i = 0; i < times.size(); i++){
				out_trc << times[i] << " " << testlls[i] << " " << starttimes[i] << " " << mergetimes[i] << " " << clusters[i] << " " << matchings[i] << std::endl;
			}
			out_trc.close();
		}
};
#define __TRACE_HPP
#endif /* __TRACE_HPP */
