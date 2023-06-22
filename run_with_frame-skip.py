import cv2
import numpy as np
import collections
import threading
import os
import time
import pycuda.driver as cuda

from utils.yolo_with_plugins import TrtYOLO
from filterpy.kalman import KalmanFilter
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

import argparse
import copy
import math
import torch 
import torch.nn as nn
from torch.autograd import Variable

import warnings

warnings.filterwarnings('ignore', category = DeprecationWarning)

s_img, s_boxes = None, None
INPUT_HW = (600, 600)
MAIN_THREAD_TIMEOUT = 20.0  # 20 sec

@jit

def arg_parse():

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    '''
    parser.add_argument("--cfg", dest = "cfgfile", help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights",  dest = "weightsfile", help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    '''
    parser.add_argument("--model",  help = 
                        "name of model(trt version)",
                        default = "yolo4-tiny-project", type = str)
    parser.add_argument("--names",  help = 
                        "namesfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "288", type = str)
    parser.add_argument("--video",help = "Input video file path",
                        default = "./video_sample/1", type = str)
    
    return parser.parse_args()




def find_distance(c1, c2):
    return int(math.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2))



# find cars that were not moving for more than a certain amount of time
def find_stopped_cars(long_car_dict,counting_frames):
    # long_stopped_cars = {}
    for ID, frames in counting_frames.items():
        if frames > 15: # this number can be changed depending on applications
            # long_stopped_cars.append(ID)
            long_car_dict[ID] = frames
    return long_car_dict



def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)  
    return o



# [x1, y1, x2, y2] -> [u, v, s, r]
def convert_bbox_to_z(bbox): # take a bbox in the form [x1,y1,x2,y2] and return z in the form [u,v,s,r] where u,v is the center of the box, s is the scale/area and r is the aspect ratio
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))




# [u, v, s, r] -> [x1, y1, x2, y2]
def convert_x_to_bbox(x, score=None): # take a bbox in the center form [x,y,s,r] and return it in the form [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))



class KalmanBoxTracker(object): # represent the internel state of individual tracked objects observed as bbox

    count = 0

    def __init__(self, bbox): # initialize a tracker using initial bbox.
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox): # update the state vector with observed bbox
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox)) # kalmanfilter.update

    def predict(self): # advance the state vector and returns the predicted bbox estimate
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0 # how many times it has been tracked consequtively
        self.time_since_update += 1 # how long it has been since the last update (update only when matched)
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self): # return the current bbox estimate.
        return convert_x_to_bbox(self.kf.x)



def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3): # assign detections to tracked object (both represented as bboxes)

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk) # det, trk: bbox

    # Hungarian Algorithm
    matched_indices = linear_assignment(-iou_matrix) # [n, 2] if iou_matrix.shape = [4,3] -> matched_indices=[3,2]
    matched_indices = np.stack(matched_indices, axis=-1)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:   
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)




def get_frame(cam, trt_yolo, conf_th, video):
    
    frame = 0
    max_age = 15
    
    trackers = []
    car_counting_frames = {}
    capture_car_list = []
    capture_car_list_stopped = []
    long_stopped_cars = {}
    
    output_file = video + '_out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv2.CAP_PROP_FPS) # 30

    output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    idstp = collections.defaultdict(list)
    
    print('\nPREDICTION STARTS\n')

    start_time = time.time() # start measuring time

    while True:
        
        ret, original_img = cam.read()
        if original_img is None:
            break
        

        # frame-skip
        if frame % 5 == 0:
            boxes, confs, clss = trt_yolo.detect(original_img, conf_th)
            past_boxes = boxes
        else: 
            boxes = past_boxes

        '''
        boxes, confs, clss = trt_yolo.detect(original_img, conf_th)
        '''

        print('frame: ', frame)

        boxes = np.array(boxes)

        H, W = original_img.shape[:2]

        trks = np.zeros((len(trackers), 5)) # trackers: previously tracked bboxes (kalmanboxtrackers)
                                            # trks: predicted bboxes based on trackers values
        to_del = []
                                            
        for t, trk in enumerate(trks): # update trks after predicting each row of trackers
            pos = trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0] # predicting each row of trks
            if np.any(np.isnan(pos)):
                to_del.append(t) # if kalmanboxtracker fails to predict
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # get rid of invalid valus such as NaN or infinites from trks
        
        for t in reversed(to_del): # erase rows with NAN
            trackers.pop(t)
        
        # boxes: [n,4], trks : [m,5]
        
        # boxes: bboxes calculated by detection
        # trks: expected location of the next bbox of the previous ones
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(boxes, trks)
        
        long_stopped_cars = find_stopped_cars(long_stopped_cars, car_counting_frames)

        for t, trk in enumerate(trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(boxes[d, :][0]) 
                xmin, ymin, xmax, ymax = boxes[d, :][0]
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)

                if find_distance(idstp[trk.id][0], [cx, cy]) < W/parts:
                    if trk.id in car_counting_frames.keys():
                        car_counting_frames[trk.id] += 1
                    else:
                        car_counting_frames[trk.id] = 1
                else:
                    if trk.id in car_counting_frames.keys():
                        print(f'{trk.id} is in car_counting_frames but not in stopped_car_IDs. Delete it from car_counting_frames')
                        car_counting_frames.pop(trk.id)
                        
                idstp[trk.id][0] = [int((xmin+xmax)/2), int((ymin+ymax)/2)]
                
                if t in long_stopped_cars.keys():
                    if t not in capture_car_list_stopped:
                        file_path = './captured_cars/captured_car_'+'{:02d}'.format(t)+'.png'
                        os.remove(file_path)
                        capture_car_list_stopped.append(t)
                    cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(original_img, 'id:'+str(trk.id)+' '+'stopped', (int(xmax)+1, int(ymax)+1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                else:
                    if t not in capture_car_list:
                        capture = original_img[ymin:ymax, xmin:xmax]
                        cv2.imwrite('./captured_cars/captured_car_'+'{:02d}'.format(t)+'.png', capture)
                        capture_car_list.append(t)
                    else:
                        file_path = './captured_cars/captured_car_'+'{:02d}'.format(t)+'.png'
                        prev_img = cv2.imread(file_path)
                        prev_img_h, prev_img_w, _ = prev_img.shape
                        if ((ymax-ymin) > prev_img_h) and ((xmax-xmin) > prev_img_w):
                            capture = original_img[ymin:ymax, xmin:xmax]
                            cv2.imwrite('./captured_cars/captured_car_'+'{:02d}'.format(t)+'.png', capture)
                    cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(original_img, 'id:' + str(trk.id), (int(xmax)+1, int(ymax)+1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    
        for k,v in long_stopped_cars.items():
            if v!= 0:
                print(f'car {k} was standing for {round(v/fps, 1)} sec, {v} frames')

        print('\n----------------------------------------\n')

        # create and initialisz new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(boxes[i, :])
            trackers.append(trk)
            trk.id = len(trackers)-1

            # new tracker id & u, v
            u, v = trk.kf.x[0], trk.kf.x[1]
            idstp[trk.id].append([u, v])

            if trk.time_since_update > max_age:
                trackers.pop(i)

        '''
        # activate it if you want to save the output video
        output_video.write(original_img) 
        '''

        resize_img = cv2.resize(original_img, (600,600))
        cv2.imshow('real-time processing', resize_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame += 1

    elapsed_time = round(time.time()-start_time, 1) # end measuring time
    FPS = round(frame/elapsed_time, 1)
    print('PREDICTION ENDED\n')
    print(f'FPS: {FPS} [frames/sec]')
    output_video.release()










if __name__ == '__main__':

    args = arg_parse()

    start = 0
    CUDA = torch.cuda.is_available()

    video_path = args.video + '.mp4'
    cam = cv2.VideoCapture(video_path)
    
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    parts = int(frame_width*0.75)
    #parts = 450

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cuda.init() # init pycuda driver

    cuda_ctx = cuda.Device(0).make_context() # GPU 0
    trt_yolo = TrtYOLO(args.model, category_num = 1)
    
    conf_th = np.array(float(args.confidence))
    get_frame(cam,trt_yolo, conf_th, args.video)
    
    del trt_yolo
    cuda_ctx.pop()
    del cuda_ctx

    cam.release()
    cv2.destroyAllWindows()