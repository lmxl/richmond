#!/usr/bin/env bash
for i in {10..13}
do
#	python richmond.py 1> report/stdout.$i.txt 2> report/stderr.$i.txt &
    python richmond_adv.py 1> report/stdout.$i.txt 2> report/stderr.$i.txt &
	sleep 15
done
