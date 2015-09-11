#!/usr/bin/env bash
for i in {0..9}
do
	python richmond.py 1> report/stdout.$i.txt 2> report/stderr.$i.txt &
	sleep 120
done