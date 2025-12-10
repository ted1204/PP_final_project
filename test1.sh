make
mkdir -p ./.middle
./seq ./testcases/testcase1.bin 0.01 300
./bin2csv ./.middle/trajectory.bin ./.middle/output.csv 3 300
python ./Generation/mp4_generator/mp4.py ./.middle/output.csv --n 3 --d 300 --t 10 --output ./Videos/t1.mp4 