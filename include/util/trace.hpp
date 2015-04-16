#ifndef __TRACE_HPP
#include<string>
#include<fstream>

class Trace{
	public:
		std::vector<double> times, objs, testlls;
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
#define __TRACE_HPP
#endif /* __TRACE_HPP */
