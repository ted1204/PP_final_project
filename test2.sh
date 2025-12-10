make
./seq ./testcases/testcase2.bin 0.01 300
./bin2csv ./.middle/trajectory.bin ./.middle/output.csv 70 300
python ./Generation/mp4_generator/mp4.py ./.middle/output.csv --n 70 --d 300 --t 10 --output ./Videos/t2.mp4 