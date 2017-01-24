import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xmltodict
import cPickle
import simplejson as json
#import progessbar

#bar = progressbar.ProgressBar()

os.chdir('/usr0/home/htpham/caffe/')
sys.path.append('/usr0/home/htpham/caffe/python')

import caffe
caffe.set_device(2)
caffe.set_mode_gpu()


def load_model():

    # load the model
    caffe_dir = '/usr0/home/rpasumar/courses/10807/caffe/'
    net = caffe.Net(caffe_dir + 'models/VGG_CNN_S/deploy.prototxt',
                    caffe_dir + 'models/VGG_CNN_S/VGG_CNN_S.caffemodel',
                    caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # note we can change the batch size on-the-fly
    # since we classify only one image, we change batch size from 10 to 1
    net.blobs['data'].reshape(1, 3, 224, 224)
    return net, transformer


def extract_fc7(image, obj, net, transformer):
    #  load the image in the data layer

    im = caffe.io.load_image(image)

    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    #  compute
    out = net.forward()

    #  fc7 layer
    global_fc7 = net.blobs['fc7'].data[0]
    obj['annotation']['fc7'] = global_fc7.tolist()

    box_objs = obj['annotation']['object']
    it = 0

    for box_obj in box_objs:
        try:
            if 'bndbox' in box_obj:
                w = box_obj['bndbox']
                box = [w['xmin'], w['ymin'], w['xmax'], w['ymax']]
                box = [int(x) for x in box]

                crop_im = np.zeros(im.shape)
                crop_im[box[0]:box[2], box[1]:box[3], :] = im[box[0]:box[2], box[1]:box[3], :]

                net.blobs['data'].data[...] = transformer.preprocess('data', crop_im)
                # compute
                out = net.forward()

                # fc7 layer
                fc7_crop = net.blobs['fc7'].data[0]
                
                box_obj['fc7'] = fc7_crop.tolist()		
	except:
            print 'issue: %d' % it


def get_boxes(image_id):
    with open('/usr0/home/rpasumar/Image Captioning/Flickr30kEntities/Annotations/'+image_id+'.xml') as f:
        obj = xmltodict.parse(f.read())

    return obj


def main():
    net, transformer = load_model()
    print 'loaded model'
		
    flickdir = '/usr0/home/rpasumar/Image Captioning/flickr30k-images'
    savedir = '/usr0/home/rpasumar/Image Captioning/flickr30k-images-fc7'
    images = os.listdir(flickdir)
     
    it = 0
    for i in range(len(images)):
        it += 1
        image = images[i]
        image_id, extn = image.split('.')
        if extn != 'jpg':
            continue
	print 'processing image: %d' % it
        obj = get_boxes(image_id)
        extract_fc7(flickdir+'/'+image, obj, net, transformer)

        with open(savedir+'/'+image_id+'.json', 'w') as f:
            json.dump(obj, f)

if __name__ == '__main__':
    main()
