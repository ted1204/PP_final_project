make
./seq ./Generation/body_generator/testcase3.bin 0.01 300
./bin2csv ./trajectory.bin ./output.csv 10000 300
python ./Generation/mp4_generator/mp4.py ./output.csv --n 10000 --d 300 --t 10 --output ./Videos/t3.mp4 