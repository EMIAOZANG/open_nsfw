# flake8: noqa
#!/usr/bin/env python
"""
Script to run inference using ResNet like models given image file or dir

Instruction:
    If you want to use this script to generate batch predictions,
    make sure you have a layer named as "prob_3" in your deploy.prototxt
"""

import numpy as np
import os
import sys
import argparse
# import glob
import time
from PIL import Image
from StringIO import StringIO
import caffe


def resize_image(data, sz=(256, 256)):
    """
    Resize image. Please use this resize logic for best results instead of the
    caffe, since it was used to generate training dataset
    :param str data:
        The image data
    :param sz tuple:
        The resized image dimensions
    :returns bytearray:
        A byte array with the resized image
    """
    img_data = str(data)
    im = Image.open(StringIO(img_data))
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = StringIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return bytearray(fh_im.read())

def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
    output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        img_data_rs = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image(StringIO(img_data_rs))

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                                            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []


def main(argv):
    # pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input file.
    parser.add_argument(
        "input_file",
        help="Path to the input image file"
    )

    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file."
    )
    parser.add_argument(
        "--output_mode",
        default="labels",
        help="output mode, can be 'labels' or 'prob' "
    )
    parser.add_argument(
        "--save_to",
        help="path to save results to"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="flag to enaable GPU for inference"
    )

    args = parser.parse_args()

    # set output path, sys.stdout default
    output_fp = sys.stdout
    if args.save_to:
        output_fp = open(args.save_to, 'a')

    # batch processing
    image_list = []
    if os.path.isdir(args.input_file):
        image_list = [os.path.join(args.input_file, x)
                      for x in os.listdir(args.input_file)
                      if os.path.splitext(x)[-1] == '.jpg']
    elif args.input_file.startswith('s3://') or args.input_file.startswith('s3n://'):
        # Adding s3 read file support
        pass
    else:
        image_list.append(args.input_file)  # assumes input file is a jpg image

    # Pre-load caffe model.
    caffe.set_mode_cpu()  # set to cpu only mode for prediction
    nsfw_net = caffe.Net(args.model_def,  # pylint: disable=invalid-name
                         args.pretrained_model, caffe.TEST)

    start_time = time.time()
    counter = 0
    for input_file in image_list:
        image_data = StringIO(open(input_file).read())

        # Load transformer
        # Note that the parameters are hard-coded for best results
        caffe_transformer = caffe.io.Transformer(
                {'data': nsfw_net.blobs['data'].data.shape})
        caffe_transformer.set_transpose('data', (2, 0, 1))   # move image channels to outermost
        caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

        # Classify.
        scores = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])

        # Scores is the array containing SFW / NSFW image probabilities
        # scores[1] indicates the NSFW probability
        if args.output_mode == 'labels':
            print >> output_fp, os.path.split(input_file)[-1] + ' ' + str(np.argmax(scores))
        else:
            print >> output_fp, os.path.split(input_file)[-1] + ' ' + ' '.join([str(x) for x in scores])
        counter += 1
        if counter % 100 == 0 and counter > 0:
            print >> sys.stdout, '{} images processed'.format(counter)
    end_time = time.time()
    time_taken = end_time - start_time
    # print system log to file
    print >> sys.stdout, "{} examples processed in {} secs, that is {} sec/imgs".format(
            len(image_list),
            time_taken, time_taken / float(len(image_list)))

    if args.save_to:
        output_fp.close()

if __name__ == '__main__':
    main(sys.argv)
