"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]

--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import cv2
import argparse

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
from time import time


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')


default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def fitLine(points, row, col):
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    left = Point(0, int(-x*vy/vx + y))
    right = Point(col-1, int((col-1-x)*vy/vx + y))
    top = Point(int(-y*vx/vy + x), 0)
    bot = Point(int((row-1-y)*vx/vy + x), row-1)

    points = [left, right, top, bot]
    points.sort(key = lambda p : (p.x - x)**2 + (p.y - y)**2)

    return points[0], points[1], [vx, vy, x, y]

def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiSeg_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
    tv_utils.set_gpus_to_use()

    # if FLAGS.input_image is None:
    #     logging.error("No input_image was given.")
    #     logging.info(
    #         "Usage: python demo.py --input_image data/test.png "
    #         "[--output_image output_image] [--logdir /path/to/weights] "
    #         "[--gpus GPUs_to_use] ")
    #     exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.    
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)
        image.set_shape([1, None, None, 3])

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    logging.info("Starting inference using {} as input".format(input_image))

    # Load and resize input image
    count = 1
    duration = 0
    video = cv2.VideoCapture("/home/zzhang52/Data/south_farm/front/VID_20170426_144800.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter("/home/zzhang52/Videos/visual_nav.avi", fourcc, 12.0, (960,540))

    while video.grab():
        start = time()
        _, image = video.retrieve()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if hypes['jitter']['reseize_image']:
            # Resize input only, if specified in hypes
            image_height = hypes['jitter']['image_height']
            image_width = hypes['jitter']['image_width']
            image = scp.misc.imresize(image, size=(image_height, image_width),
                                      interp='bilinear')

        # Run KittiSeg model on image
        feed = {image_pl: image}
        softmax = prediction['softmax']
        output = sess.run([softmax], feed_dict=feed)
        duration += time() - start
        count += 1

    # Reshape output from flat vector to 2D Image
        shape = image.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

        # Plot confidences as red-blue overlay
        # rb_image = seg.make_overlay(image, output_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a `hard` prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold

        thresh = (255 * street_prediction).astype(np.uint8)

        _, thresh = cv2.threshold(thresh, 100, 255, 0)

        _, contours, _ = cv2.findContours(thresh, 
            mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
        
        contours.sort(key = cv2.contourArea, reverse = True)
        mask = np.zeros(shape, np.uint8)
        cv2.drawContours(mask, contours, 0, (0, 255, 0), -1)

        cnt = np.asarray(contours[0])
        cnt = np.squeeze(cnt)
        cnt = cnt[np.argsort(cnt[:,1])]
        cnt = np.expand_dims(cnt, axis = 1)

        points = []
        idx = 0
        x_sum = 0
        x_count = 0

        for i in range(shape[0]):
            while idx < cnt.shape[0] and cnt[idx][0][1] == i:
                x_sum += float(cnt[idx][0][0])
                x_count += 1
                idx += 1

            if x_count != 0:
                avg = int(x_sum / x_count)
                points.append(np.array([[avg, i]]))
                x_sum = 0
                x_count = 0

        points = np.asarray(points)

        # p1, p2, [vx,vy,x,y] = fitLine(points, shape[0], shape[1])

        # [vx,vy,x,y] = cv2.fitLine(points, cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((shape[1]-x)*vy/vx)+y)

        p1 = Point(points[0][0][0], points[0][0][1])
        p2 = Point(points[-1][0][0], points[-1][0][1])

        dx = float(p1.x - p2.x)
        dy = float(p1.y - p2.y)
        r = np.arctan(dx / dy) * 180 / np.pi
        t = float(shape[1]/2 - p2.x) / shape[1]
        # r = np.sign(r) * (90 - abs(r))
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        alpha = 0.3
        overlay = cv2.addWeighted(image, 1-alpha, mask, alpha, 0)

        # Plot the hard prediction as green overlay
        # green_image = tv_utils.fast_overlay(image, street_prediction)
        # green_image = cv2.cvtColor(green_image, cv2.COLOR_RGB2BGR)
        cv2.line(overlay,(p1.x, p1.y),(p2.x, p2.y),(0,0,200),2)
        cv2.putText(overlay, "Heading: {:.2f}".format(r), (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(overlay, "Offset: {:.3e}".format(t), (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("show", overlay)
        writer.write(overlay)
        if cv2.waitKey(15) == 27:
            break

    cv2.destroyAllWindows()
    video.release()
    writer.release()

    # Save output images to disk.
    # if FLAGS.output_image is None:
    #     output_base_name = input_image
    # else:
    #     output_base_name = FLAGS.output_image

    # raw_image_name = output_base_name.split('.')[0] + '_raw.png'
    # rb_image_name = output_base_name.split('.')[0] + '_rb.png'
    # green_image_name = output_base_name.split('.')[0] + '_green.png'

    # scp.misc.imsave(raw_image_name, street_prediction)
    # scp.misc.imsave(rb_image_name, rb_image)
    # scp.misc.imsave(green_image_name, green_image)

    # logging.info("")
    # logging.info("Raw output image has been saved to: {}".format(
    #     os.path.realpath(raw_image_name)))
    # logging.info("Red-Blue overlay of confs have been saved to: {}".format(
    #     os.path.realpath(rb_image_name)))
    # logging.info("Green plot of predictions have been saved to: {}".format(
    #     os.path.realpath(green_image_name)))

    print ("FPS: {}".format(count / duration))

if __name__ == '__main__':
    tf.app.run()
