"""ssd.py

This module implements the TrtSSD class.
"""


import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

from darknet import Darknet
import torch 
import torch.nn as nn
from torch.autograd import Variable



def _preprocess_trt(img, shape=(288, 288)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img *= (2.0/255.0)
    img -= 1.0
    return img


def _postprocess_trt(img, outs, conf_th):
    # Process the output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_th:
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                width = int(detection[2] * img.shape[1])
                height = int(detection[3] * img.shape[0])
                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = int(center_x + width / 2)
                y2 = int(center_y + height / 2)
                if class_id == 0:
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x1, y1, x2, y2])
    return boxes, confidences, class_ids


class TrtYOLO_thread(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""

    #def _load_plugins(self):
        #if trt.__version__[0] < '7':
            #ctypes.CDLL("ssd/libflattenconcat.so")
        #trt.init_libnvinfer_plugins(self.trt_logger, '')

    #def _load_engine(self):
        #TRTbin = 'ssd/TRT_%s.bin' % self.model
        #with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            #return runtime.deserialize_cuda_engine(f.read())

    '''
    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings
        '''

    def __init__(self, net):
        """Initialize TensorRT plugins, engine and conetxt."""
        #net = cv2.dnn.readNetFromDarknet(args.cfgfile, args.weightsfile)
        
        #if torch.cuda.is_available():
            #net.cuda()
        #frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        #frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #fps = video.get(cv2.CAP_PROP_FPS)
        self.net = net
        #self.frame_width = frame_width
        #self.frame_height = frame_height
        #self.fps = fps
        #self.confidence_th = args.confidence
        #self.cuda_ctx = cuda_ctx
        #if self.cuda_ctx:
            #self.cuda_ctx.push()

        #self.trt_logger = trt.Logger(trt.Logger.INFO)
        #self._load_plugins()
        #self.engine = self._load_engine()

        #try:
            #self.context = self.engine.create_execution_context()
            #self.stream = cuda.Stream()
            #self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        #except Exception as e:
            #raise RuntimeError('fail to allocate CUDA resources') from e
        #finally:
            #if self.cuda_ctx:
                #self.cuda_ctx.pop()

    #def __del__(self):
        """Free CUDA memories and context."""
        #del self.cuda_outputs
        #del self.cuda_inputs
        #del self.stream

    def detect(self, img, conf_th=0.3):
        net = self.net
        #net.eval()
        """Detect objects in the input image."""
        blob = cv2.dnn.blobFromImage(img, 1/255, (288, 288), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)
        return _postprocess_trt(img, outs, conf_th)
