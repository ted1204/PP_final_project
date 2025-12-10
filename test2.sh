make
./seq ./Generation/body_generator/testcase2.bin 0.01 300
./bin2csv ./trajectory.bin ./output.csv 70 300
python ./Generation/mp4_generator/mp4.py ./output.csv --n 70 --d 300 --t 10 --output ./Videos/t2.mp4 