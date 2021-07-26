import time
from cyclegan.options.train_options import TrainOptions
from cyclegan.data import create_dataset
from cyclegan.models import create_model
import pickle
import numpy as np

def ranking(dic1,dic2):
    dic={}
    reverse={}
    values=[]
    for element in list(dic1.keys()):
        value=dic1[element]/dic2[element]
        dic[element]=value
        reverse[value]=element
        values.append(value)
    
    values.sort()
    print(reverse[values[-1]]+','+str(values[-1]))
    print(reverse[values[-2]]+','+str(values[-2]))
    print(reverse[values[-3]]+','+str(values[-3]))
    print(reverse[values[-4]]+','+str(values[-4]))
    print(reverse[values[-5]]+','+str(values[-5]))
    print(reverse[values[-6]]+','+str(values[-6]))
    print(reverse[values[-7]]+','+str(values[-7]))
    print(reverse[values[-8]]+','+str(values[-8]))
    print(reverse[values[-9]]+','+str(values[-9]))
    print(reverse[values[-10]]+','+str(values[-10]))
    print(reverse[values[-11]]+','+str(values[-11]))
    print(reverse[values[-12]]+','+str(values[-12]))
    print(reverse[values[-13]]+','+str(values[-13]))
    print(reverse[values[-14]]+','+str(values[-14]))
    print('..')
    print(reverse[values[5]]+','+str(values[5]))
    print(reverse[values[4]]+','+str(values[4]))
    print(reverse[values[3]]+','+str(values[3]))
    print(reverse[values[2]]+','+str(values[2]))
    print(reverse[values[1]]+','+str(values[1]))

if __name__ == '__main__':
    images_ranking={}
    images_number={}
    opt = TrainOptions().parse()# get training options

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        print(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights



            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            if data['B_paths'][0][23:] in list(images_ranking.keys()):
                images_ranking[data['B_paths'][0][23:]]+=model.loss_cycle_B()
                images_number[data['B_paths'][0][23:]]+=1
            else:
                images_ranking[data['B_paths'][0][23:]]=model.loss_cycle_B()
                images_number[data['B_paths'][0][23:]]=1
        print(ranking(images_ranking,images_number))
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
