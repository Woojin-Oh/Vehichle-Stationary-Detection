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

warnings.filterwarnings("ignore", category=DeprecationWarning)

s_img, s_boxes = None, None
INPUT_HW = (600, 600)
MAIN_THREAD_TIMEOUT = 200.0  # 200 seconds

# SORT Multi object tracking

#iou
@jit

def arg_parse():
    """
    detect module에 대한 인자 분석하기
    
    """
    
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
    #long_stopped_cars = {}
    for ID, frames in counting_frames.items():
        if frames > 15: # this number can be changed to increase work efficiency
            #long_stopped_cars.append(ID)
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
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

#[x1, y1, x2, y2] -> [u, v, s, r]
def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [u,v,s,r] where u,v is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

#[u, v, s, r] -> [x1, y1, x2, y2]
def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

#
class KalmanBoxTracker(object): #object - 생략 가능
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
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

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox)) #kalmanfilter.update

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0 #연속으로 몇번 track 됐는지
        self.time_since_update += 1 #마지막 update로부터 얼마나 흘렀는지, match 할때만 업데이트
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)#det, trk: bbox

    #Hungarian Algorithm
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
        matches = np.concatenate(matches, axis=0) #matches: list, [1,2]가 여러개 있는 리스트, 이거를 [n,2] 넘파이 배열로 만듬

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Inference_Thread(threading.Thread):
    def __init__(self, condition, cam, args):
        """__init__

        # Arguments
            condition: the condition variable used to notify main
                       thread about new frame and detection result
            cam: the camera object for reading input image frames
            model: a string, specifying the TRT SSD model
            conf_th: confidence threshold for detection
        """
        threading.Thread.__init__(self)
        self.condition = condition
        self.cam = cam
        self.conf_th = np.array(float(args.confidence))
        #self.net = cv2.dnn.readNet(args.cfgfile, args.weightsfile)
        #if torch.cuda.is_available():
            #self.net.cuda()
        self.cuda_ctx = None  # to be created when run
        self.trt_ssd = None   # to be created when run
        self.running = False

    def run(self):
        global s_img, s_boxes

        print('Inference_Thread: loading the TRT YOLO engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_yolo = TrtYOLO(args.model, category_num = 1)
        print('Inference_Thread: start running...')
        self.running = True
        frame = 0
        while self.running:
            frame += 1
            start_child = time.time()
            ret, origin_img = self.cam.read()
            if origin_img is None:
                break
            #img = cv2.resize(origin_img, (416, 416))
            if frame % 5 == 1:
                boxes, confs, clss = self.trt_yolo.detect(origin_img, self.conf_th)
                prev_boxes = boxes
            else:
                boxes = prev_boxes
            with self.condition:
                s_img, s_boxes = origin_img, boxes
                self.condition.notify()
            #prev_boxes = boxes
            end_child = time.time()
            print('child thread time: ', end_child - start_child)
            sleep_time = 0.07 - round((end_child - start_child), 2)
            time.sleep(max(sleep_time, 0))
        del self.trt_yolo
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('Inference_Thread: stopped...')
        
        

    def stop(self):
        print('here3')
        self.running = False
        self.join()


def get_frame(condition, cam, video, parts):
    frame = 0
    max_age = 15
    
    trackers = [] #KalmanBoxTracker(boxes[i, :]) 추가됨, KBT class가 원소인 리스트, 여태까지 봤던 모든 bbox들 저장, tracking
    car_counting_frames = {}
    capture_car_list = []
    capture_car_list_stopped = []
    long_stopped_cars = {}
    

    output_file = video + '_out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv2.CAP_PROP_FPS) # 25

    #output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    idstp = collections.defaultdict(list)
    
  
    while True:
        with condition:
            if condition.wait(timeout=MAIN_THREAD_TIMEOUT):  #condition.notify일때까지 기다림 
                original_img, boxes = s_img, s_boxes
                start = time.time()
            else:
                break

        frame += 1
        print('frame: ', frame)
        
        boxes = np.array(boxes)

        H, W = original_img.shape[:2]

        trks = np.zeros((len(trackers), 5))   #trackers: 이전에 track하던 bbox들(kalmanboxtracker), trks: trackers 값 기반으로 predict한 bbox들
        to_del = []

        for t, trk in enumerate(trks): # t: t번째 행 index, trk: t번째 행 , trackers의 각 행을 predict한 후, trks 업데이트
            pos = trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0] #trks의 각 행 predict
            if np.any(np.isnan(pos)):
                to_del.append(t) #만약, kalmanboxtracker가 predict 실패할 경우
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) #NaN이나 infinites와 같은 유효하지 않은 값들 trks에서 제외
        for t in reversed(to_del): #NAN 값이 있는 행 제외
            trackers.pop(t)
        
        #boxes: [n,4], trks : [m,5]
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(boxes, trks)  #boxes: detection을 통해 계산한 bbox들, trks: 이전 bbox들의 다음 bbox 예상 위치
        #matched, unmatced_dets, unmatched_trks: 해당 인덱스들을 ndarray로 저장 [a,2](첫번째 열: det_index, 두번째열: trk_index), [k,1], [j,1]

#______________________ 여기부터 in, out 설정

        long_stopped_cars = find_stopped_cars(long_stopped_cars, car_counting_frames)

        for t, trk in enumerate(trackers): #trk - type: kalmanboxtracker
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(boxes[d, :][0]) #boxes[d,:] : [f, 4] , 여기서 f는 해당 조건 만족하는 인덱스 개수
                xmin, ymin, xmax, ymax = boxes[d, :][0] #boxes: 예측한 bbox
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)

                #idstp: 여태 track하던 id들 dicktionary, idstp[trk.id] = [u,v] , u, v: center bbox 좌표
                #idstp[trk.id][0][1] = v, 해당 tracking bbox의 가장 첫 bbox, cy: 현재 detect한 bbox
                if find_distance(idstp[trk.id][0], [cx, cy]) < W/parts:
                    if trk.id in car_counting_frames.keys():
                        car_counting_frames[trk.id] += 1
                    # Adding a new car ID
                    else:
                        car_counting_frames[trk.id] = 1
                else:
                    if trk.id in car_counting_frames.keys():
                        print(f"{trk.id} is in car_counting_frames but is not in stopped_car_IDs - delete it from car_counting_frames")
                        car_counting_frames.pop(trk.id)
                
                idstp[trk.id][0] = [int((xmin + xmax) / 2),int((ymin + ymax) / 2)]


                if t in long_stopped_cars.keys():
                    if t not in capture_car_list_stopped:
                        file_path = './captured_cars/captured_car_'+'{:02d}'.format(t)+'.png'
                        os.remove(file_path)
                        capture_car_list_stopped.append(t)
                    cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(original_img, "id:" + str(trk.id) + " stopped", (int(xmax) + 1, int(ymax) + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2) #0.5: 글자 크기
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
                    cv2.putText(original_img, "id:" + str(trk.id), (int(xmax) + 1, int(ymax) + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2) #0.5: 글자 크기

        for k,v in long_stopped_cars.items():
            if v!= 0:
                print(f"car {k} was standing for {v/fps} second, {v} frames")
        print('______________________')
            
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(boxes[i, :])
            trackers.append(trk)
            trk.id = len(trackers) -1

            #new tracker id & u, v
            u, v = trk.kf.x[0], trk.kf.x[1]
            idstp[trk.id].append([u, v])

            #??? trk는 여기서 처음 선언되는데, trk.time_since_update가 왜 있지? 왜 pop(i) ? 
            if trk.time_since_update > max_age: #max_age = 15, 15프레임동안 update(match)되지 않으면, 그 tracking bbox는 삭제 
                trackers.pop(i)

        #output_video.write(original_img)

        resize_img = cv2.resize(original_img, (1080,1080))
        cv2.imshow("dst",resize_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end = time.time()
        print('main_thread 시간: ', (end-start))
        
    #output_video.release()

if __name__ == '__main__':

    
    args = arg_parse()
    CUDA = torch.cuda.is_available()
    
    video_path = args.video + '.mp4'
    cam = cv2.VideoCapture(video_path)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    parts = int(frame_width*0.75)
    #parts = 300

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cuda.init()  # init pycuda driver

    condition = threading.Condition()
    trt_thread = Inference_Thread(condition, cam, args)
    trt_thread.start()  # start the child thread

    get_frame(condition, cam, args.video, parts)
    
    trt_thread.stop()   # stop the child thread

    cam.release()
    cv2.destroyAllWindows()
