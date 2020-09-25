#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
import tensorflow as tf  
import time



def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def main(dataset, opt, save_images=True):
    # log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args(["--model", "/disk/projectEyes/GazeCorrection/OpenVinoGazeCorrection/inference_graph_batch1/inference_graph_3_batch1.xml", "--input", "/disk/projectEyes/dataSet/NewGazeData/0"])
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU" ) #"CPU" OR "MYRIAD"
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    # assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    # assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    print(net.inputs['Placeholder'].shape)
    # input_blob = next(iter(net.input_info))
    # out_blob = next(iter(net.outputs))
    # net.batch_size = len(args.input)

    # # Read and pre-process input images
    # n, c, h, w = net.input_info[input_blob].input_data.shape
    # images = np.ndarray(shape=(n, c, h, w))
    # for i in range(n):
    #     image = cv2.imread(args.input[i])
    #     if image.shape[:-1] != (h, w):
    #         log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
    #         image = cv2.resize(image, (w, h))
    #     image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    #     images[i] = image
    # log.info("Batch size is {}".format(n))

    ######################
    # Loading initialized variables for input from file:
    #######################
    # with open('/disk/nGraph/openvino/inference-engine/ie_bridges/python/sample/classification_sample/array_vars/placeholder', 'rb') as file1:
    #     placeholder_eye_left_coord_batch = np.load(file1)
    # with open('/disk/nGraph/openvino/inference-engine/ie_bridges/python/sample/classification_sample/array_vars/placeholder_1', 'rb') as file1:
    #     placeholder_eye_right_coord_batch = np.load(file1)
    # with open('/disk/nGraph/openvino/inference-engine/ie_bridges/python/sample/classification_sample/array_vars/placeholder_2', 'rb') as file1:
    #     placeholder_image_batch = np.load(file1)
    # with open('/disk/nGraph/openvino/inference-engine/ie_bridges/python/sample/classification_sample/array_vars/placeholder_3', 'rb') as file1:
    #     placeholder_image_wd_eyes_masked_batch = np.load(file1)
    
    # placeholder_image_batch = np.transpose(np.array(placeholder_image_batch), axes=[0, 3, 1, 2])
    # placeholder_image_wd_eyes_masked_batch = np.transpose(np.array(placeholder_image_wd_eyes_masked_batch), axes=[0, 3, 1, 2])

    # # Loading model to the plugin
    # log.info("Loading model to the plugin")
    # exec_net = ie.load_network(network=net, device_name=args.device)

    # # Start sync inference
    # log.info("Starting inference in synchronous mode")
    # # res = exec_net.infer(inputs={input_blob: images})
    # res = exec_net.infer(inputs={'Placeholder': placeholder_eye_left_coord_batch, 'Placeholder_1': placeholder_eye_right_coord_batch, 'Placeholder_2' : placeholder_image_batch, 'Placeholder_3' : placeholder_image_wd_eyes_masked_batch})

    ###################################################
    # USING TENSORFLOW AND DATASET OBJECT TO FORM INPUTS FOR INFERENCE -- ANANT
    ###################################################
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # self.saver = tf.train.Saver()
    
    #############
    # Loading model to the plugin
    #############
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    with tf.Session(config=config) as sess:
        sess.run(init)
        _,_,_, testbatch, testmask = dataset.input()
        #testbatch, testmask = self.dataset.custom_test_input()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        batch_num = 3451 / opt.batch_size
        # batch_num = 13 #Have made batch size = 1 to skip generation of 
        #################################
        # STARTING TIMING 
        ##################################
        start_time = time.time() 

        for j in range(int(batch_num)):
            real_test_batch, real_eye_pos = sess.run([testbatch, testmask])
            batch_masks, batch_left_eye_pos, batch_right_eye_pos = get_Mask_and_pos(real_eye_pos, opt)
            real_test_batch_transposed = np.transpose(np.array(real_test_batch), axes = [0,3,1,2])
            batch_masks = np.transpose(np.array(batch_masks), axes = [0,3,1,2])
            # output = sess.run([self.x, self.y], feed_dict=f_d)

            # Start sync inference
            log.info("Starting inference in synchronous mode")
            # res = exec_net.infer(inputs={input_blob: images})
            res = exec_net.infer(inputs={'Placeholder': batch_left_eye_pos, 'Placeholder_1': batch_right_eye_pos, 'Placeholder_2' : real_test_batch_transposed, 'Placeholder_3' : batch_masks})
            if save_images == True:
                if j % 100 == 0 : #Considering the batch_num is 0 #IF IMAGES ARE NEEDED TO BE SAVED PERIODICALLY, tab THE CODE BELOW.
                    ######################
                    # IF ONLY RESULTANT IMAGE NEEDS TO BE SAVED W/O CONCATINATION: 
                    # -ANANT
                    ######################
                    # res_transposed = np.transpose(np.array(res['add']), axes=[0,2,3,1])
                    # output_image = np.reshape(res_transposed, [256, 256, 3])
                    # imageio.imwrite(image_path, output_image)

                    ######################
                    # IF CONCAT OF INPUT + OUTPUT NEEDS TO BE SAVED:
                    # - ANANT
                    ######################            
                    res_transposed = np.transpose(np.array(res['add']), axes=[0,2,3,1])
                    output_concat = Transpose([real_test_batch, res_transposed])
                    image_path = '{}/{:02d}.jpg'.format("/disk/projectEyes/GazeCorrection/OpenVinoGazeCorrection/outputTestImages", j)
                    import imageio
                    imageio.imwrite(image_path, output_concat)
                    # save_images(output_image, '{}/out_1st.jpg'.format("/disk/nGraph/openvino/inference-engine/ie_bridges/python/sample/classification_sample/output"))
                    print('DONE')
        #################################
        # ENDING TIMING 
        ##################################
        print("\n \n INNER Time elapsed in GazeGan inference using OV of 3451 images = ", time.time() - start_time )
        
        coord.request_stop()
        coord.join(threads)


def Transpose(list):

        refined_list = np.transpose(np.array(list), axes=[1, 2, 0, 3, 4])
        refined_list = np.reshape(refined_list, [refined_list.shape[0] * refined_list.shape[1],
                                                 refined_list.shape[2] * refined_list.shape[3], -1])
        return refined_list


def get_Mask_and_pos(eye_pos, opt, flag=0):
    eye_pos = eye_pos
    batch_mask = []
    batch_left_eye_pos = []
    batch_right_eye_pos = []
    for i in range(opt.batch_size):

        current_eye_pos = eye_pos[i]
        left_eye_pos = []
        right_eye_pos = []

        if flag == 0:

            mask = np.zeros(shape=[opt.img_size, opt.img_size, opt.output_nc])
            scale = current_eye_pos[1] - 15
            down_scale = current_eye_pos[1] + 15
            l1_1 =int(scale)
            u1_1 =int(down_scale)
            #x
            scale = current_eye_pos[0] - 25
            down_scale = current_eye_pos[0] + 25
            l1_2 = int(scale)
            u1_2 = int(down_scale)

            mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
            left_eye_pos.append(float(l1_1)/opt.img_size)
            left_eye_pos.append(float(l1_2)/opt.img_size)
            left_eye_pos.append(float(u1_1)/opt.img_size)
            left_eye_pos.append(float(u1_2)/opt.img_size)

            scale = current_eye_pos[3] - 15
            down_scale = current_eye_pos[3] + 15
            l2_1 = int(scale)
            u2_1 = int(down_scale)

            scale = current_eye_pos[2] - 25
            down_scale = current_eye_pos[2] + 25
            l2_2 = int(scale)
            u2_2 = int(down_scale)

            mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

            right_eye_pos.append(float(l2_1) / opt.img_size)
            right_eye_pos.append(float(l2_2) / opt.img_size)
            right_eye_pos.append(float(u2_1) / opt.img_size)
            right_eye_pos.append(float(u2_2) / opt.img_size)

        batch_mask.append(mask)
        batch_left_eye_pos.append(left_eye_pos)
        batch_right_eye_pos.append(right_eye_pos)

    return np.array(batch_mask), np.array(batch_left_eye_pos), np.array(batch_right_eye_pos)
