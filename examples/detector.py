#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
from os.path import splitext

from NumPyNet.parser import data_config
from NumPyNet.parser import get_labels

from NumPyNet.image import Image

from NumPyNet.network import Network

__author__  = ['Mattia Ceccarelli', 'Nico Curti']
__email__   = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


def parse_args ():

  description = 'NumPyNet Yolov3 detector test example'

  parser = argparse.ArgumentParser(description = description)
  parser.add_argument('--data',
                      dest='data_cfg',
                      required=True,
                      type=str,
                      action='store',
                      help='Data cfg filename'
                      )
  parser.add_argument('--cfg',
                      dest='netcfg',
                      required=False,
                      type=str,
                      action='store',
                      help='Configuration network filename/path',
                      default=''
                      )
  parser.add_argument('--weights',
                      dest='weights',
                      required=False,
                      type=str,
                      action='store',
                      help='Weights filename/path',
                      default='')
  parser.add_argument('--names',
                      dest='namesfile',
                      required=False,
                      type=str,
                      action='store',
                      help='Labels (names) filename/path',
                      default=''
                      )
  parser.add_argument('--input',
                      dest='input',
                      required=False,
                      type=str,
                      action='store',
                      help='Input filename/path',
                      default=''
                      )
  parser.add_argument('--output',
                      dest='outfile',
                      required=False,
                      type=str,
                      action='store',
                      help='Output filename/path',
                      default=''
                      )
  parser.add_argument('--thresh',
                      dest='thresh',
                      required=False,
                      type=float,
                      action='store',
                      help='Probability threshold prediction',
                      default=.5
                      )
  parser.add_argument('--hierthresh',
                      dest='hier',
                      required=False,
                      type=float,
                      action='store',
                      help='Hierarchical threshold',
                      default=.5
                      )
  parser.add_argument('--fullscreen',
                      dest='fullscreen',
                      required=False,
                      type=bool,
                      action='store',
                      help='Fullscreen bool option',
                      default=False
                      )
  parser.add_argument('--classes',
                      dest='classes',
                      required=False,
                      type=int,
                      action='store',
                      help='Number of classes to read',
                      default=-1
                      )
  parser.add_argument('--nms',
                      dest='nms',
                      required=False,
                      type=float,
                      action='store',
                      help='Threshold of IOU value',
                      default=.45
                      )
  parser.add_argument('--save',
                      dest='save',
                      required=False,
                      type=bool,
                      action='store',
                      help='Save the image with detection boxes',
                      default=False
                      )

  args = parser.parse_args()

  return args


def main():

  done = True
  args = parse_args()

  data_cfg = data_config(args.data_cfg)

  args.netcfg    = data_cfg.get('cfg',       default=args.netcfg)
  args.weights   = data_cfg.get('weights',   default=args.weights)
  args.namesfile = data_cfg.get('names',     default=args.namesfile)

  if not args.netcfg or not args.weights:
    raise ValueError('Network config AND network weights must be given')

  # names = get_labels(args.namesfile, args.classes)

  net = Network(batch=32)
  net.load(args.netcfg, args.weights)

  net_w, net_h, _ = net.input_shape

  if not args.input:
    args.input = input('Enter Image Path: ')
    done = False if not args.input else True

  # set the output filename
  args.outfile = args.outfile if args.outfile else splitext(args.input)[0] + '_detected'


  while done:

    # load image from file
    input_image = Image(filename=args.input)

    # pad-resize image if it is necessary
    input_image = input_image.letterbox(net_dim=(net_w, net_h)) if input_image.shape[:2] != net.shape else input_image

    _ = net.predict(X=input_image)

    # insert boxes evaluation and draw-detection and show image

    input_image.show(window_name=outfile, ms=0, fullscreen=fullscreen)

    if args.save:
      input_image.save(filename=outfile)


    args.input = input('Enter Image Path: ')
    done = False if not args.input else True




if __name__ == '__main__':

  main ()
