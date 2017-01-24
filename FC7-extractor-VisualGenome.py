import sys
import os
from PIL import Image as PIL_Image
import requests
from StringIO import StringIO
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from urllib import urlretrieve
import simplejson as json
#import progessbar

#bar = progressbar.ProgressBar()

os.chdir('/usr0/home/htpham/caffe/')
sys.path.append('/usr0/home/htpham/caffe/python')

import caffe
caffe.set_device(2)
caffe.set_mode_gpu()

#import VisualGenomeDriver
sys.path.append('/usr0/home/rpasumar/Image-Captioning/VisualGenome/visual_genome_python_driver/')
from src import api

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

def extract_fc7_visualgenome(image, regions,  net, transformer):
    #response = requests.get(image.url)
    #im = PIL_Image.open(StringIO(response.content))
    urlretrieve(image.url, '/usr0/home/rpasumar/Image-Captioning/src/img.jpg')
    im = caffe.io.load_image('/usr0/home/rpasumar/Image-Captioning/src/img.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    #  compute
    out = net.forward()

    #  fc7 layer
    global_fc7 = net.blobs['fc7'].data[0]
    region_fc7s = list()
    for region in regions:
        # x, y are top left coordinates
        xmin = int(region.x)
        ymax = int(region.y)
        xmax = xmin + int(region.width)
        ymin = ymax - int(region.height)
        crop_im = np.zeros(im.shape)
        crop_im[xmin:xmax, ymin:ymax, :] = im[xmin:xmax, ymin:ymax, :]
        net.blobs['data'].data[...] = transformer.preprocess('data', crop_im)
        # compute
        out = net.forward()
        # fc7 layer
        fc7_crop = net.blobs['fc7'].data[0]
        region_fc7s.append(fc7_crop.tolist())
    return global_fc7.tolist(), region_fc7s

def main():
    net, transformer = load_model()
    print 'loaded model'

    # load VisualGenomeDriver
    savedir = '/usr0/home/rpasumar/Image-Captioning/VisualGenome-fc7/' 
    output_dict = dict()
    
    ids = api.GetAllImageIds()
    #ids = api.GetImageIdsInRange(startIndex=2000, endIndex=2001)
    count = 0
    for idx in ids:
        print 'processing id: %d' %idx
        image = api.GetImageData(id=idx)
        regions = api.GetRegionDescriptionsOfImage(id=idx)
        # extract_fc7_visualgenome returns a dict with keys global_fc7 and region_fc7s
        global_fc7, region_fc7s = extract_fc7_visualgenome(image, regions, net, transformer) 
        output_dict[idx] = {'global_fc7': global_fc7, 'regional_fc7s': region_fc7s}
        with open(savedir+'/'+str(idx)+'.json','w') as fw:
            json.dump(output_dict, fw)

if __name__ == '__main__':
    main()
