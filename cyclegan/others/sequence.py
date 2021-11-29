import os


for i in range(55,155,10):
    os.system("python3 test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --num_test 500 --epoch "+str(i)+" --results_dir "+str(i))
