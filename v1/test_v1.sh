make
./seq_v1 100 0.0001 2000
./bin2csv ./trajectory.bin ./output.csv 100 2000
python ../Generation/mp4_generator/mp4.py ./output.csv --n 100 --d 2000 --t 10 --output out.mp4 