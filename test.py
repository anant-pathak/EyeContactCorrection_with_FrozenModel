from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from Dataset import Dataset
from GazeGAN import Gaze_GAN
from OpenVinoGazeCorrection import openVinoGazeGan
from config.train_options import TrainOptions



opt = TrainOptions().parse()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    dataset = Dataset(opt)
    start_time = time.time()
    # gaze_gan = Gaze_GAN(dataset, opt)
    # gaze_gan.build_test_model()
    # gaze_gan.test(freeze_model = False, flag_save_images = True)
    # print("\n \n OUTER Time elapsed in GazeGan inference using TF of 3451 images = ", time.time() - start_time )
    #start_time = time.time()
    openVinoGazeGan.main(dataset, opt, save_images = False)
    print("\n \n OUTER Time elapsed in OV inference of 3451 images = ", time.time() - start_time )
    print("\n \n Done both inferences")
    
