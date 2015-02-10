#!/bin/bash
if [ "$(whoami)" == "root" ] 
then
	echo "Installing files to /usr/local/include/sdabnp/..."
	rm -rf /usr/local/include/sdabnp
	mkdir -p /usr/local/include/sdabnp
	cp -r include/* /usr/local/include/sdabnp/
else
	echo "Root priviledges required."
fi

