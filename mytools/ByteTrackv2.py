#nuScenes数据集
import pandas as pd 
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.render import visualize_sample, dist_pr_curve
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox

import numpy as np
from numpy import *
import json
from pyquaternion import Quaternion
import copy

#可视化
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#算法函数
import mytools.math_algo as ma
from mytools.Kalman import KalmanFilter
import mytools.covariance as covariance

#路径以及目标
TRACKING_PATH              = "data/tracking/"
TRAINVAL_PATH              = "data/trainval/"
TEST_PATH                  = "data/testval/"
TRAINVAL_PATH_MINI         = "data/trainval_mini/"

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
  ]

DT = 0.5#时间间隔

def get_sample_token_sequence(nusc):
    #获取每个scene的所有token
    sample_token_sequences = []
    for scene in nusc.scene:
        sample_token_sequence = []
        sample_token = scene['first_sample_token']
        while sample_token:
            sample_token_sequence.append(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
        sample_token_sequences.append(sample_token_sequence)
    return sample_token_sequences

#关联算法
def association(Detections, Tracklets, hung_thresh):
    if len(Detections) == 0 or len(Tracklets) == 0:
        similarity_matrix = [[]]#首次关联部分
    else:
        #构建相似矩阵
        similarity_matrix = []
        for detection in Detections:
            similarity_row = []
            for tracklet in Tracklets:
                if tracklet['Found'] == False:
                    back_pred = backward_prediction(detection) # 使用当前帧的速度预测上一帧的位置
                    similarity_row.append(1-(1+ma.GIoU(back_pred, tracklet))/2) 
                else:
                    similarity_row.append(np.inf) # 跳过已经找到的tracklet
            similarity_matrix.append(similarity_row)
    matches, unmatched_det_index, _ = ma.hungarian_algorithm(np.array(similarity_matrix), thresh=hung_thresh) # 使用匈牙利算法进行关联
    unmatched_detections = []
    for matched_pair in matches: # 更新tracklet的状态
        det = Detections[matched_pair[0]]
        track = Tracklets[matched_pair[1]]
        velocity = det["velocity"]
        det_state = format_det_state(det)
        track['state'] = det_state
        track['Found'] = True 
        track['nb_lost_frame'] = 0 #因为已经找到，重置nb_lost_frame
        track['kalman'].update(det_state)
        track['velocity'] = velocity
        track['tracking_score'] = det["detection_score"]
    if len(Detections) == 0:
        unmatched_detections = []
    else:
        for unmatched_det in unmatched_det_index:
            unmatched_detections.append(Detections[unmatched_det])

    return unmatched_detections


def backward_prediction(detection):  
    #后向预测
    back_pred = format_det_state(detection)
    back_pred[0] = back_pred[0] - back_pred[-3]*DT
    back_pred[1] = back_pred[1] - back_pred[-2]*DT
    return back_pred 


def format_det_state(det):
    #将detection转换为state
    _, _, orientation = ma.euler_from_quaternion(det["rotation"][1], det["rotation"][2], det["rotation"][3], det["rotation"][0])
    
    state = [det["translation"][0], det["translation"][1], det["translation"][2], \
            det["size"][0],        det["size"][1],        det["size"][2],      orientation, \
            det["velocity"][0],    det["velocity"][1],    0]
    return state



def commonelems(x, y):
    common_values = [value for value in x if value in y]
    return bool(common_values)

# def commonelems(x,y):
#     common=False
#     for value in x:
#         if value in y:
#             common=True
#             break
#     return common

def get_common_tokens(token_list, pred):
    val_token_list = []
    for scene_tokens in token_list:
        if commonelems(scene_tokens, list(pred.keys())):
            val_token_list.append(scene_tokens)
    return val_token_list


def format_to_nuscene(sample_token, tracks):#将track转换为nuscene格式
  rotation = ma.quaternion_from_euler(0,0,tracks['state'][6])
  
  sample_result = {
    'sample_token': sample_token,
    'translation': [tracks['state'][0], tracks['state'][1], tracks['state'][2]],
    'size': [tracks['state'][3], tracks['state'][4], tracks['state'][5]],
    'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
    'velocity': [0, 0], 
    'tracking_id': str(int(tracks['track_id'])),
    'tracking_name': tracks['tracking_name'],
    'tracking_score': tracks['tracking_score']
  }

  return sample_result


def load_data(dataset=None, detection_path=None, eval_split=None, verbose=True):
    #加载数据
    if dataset =="trainval":
        nusc = NuScenes(version='v1.0-trainval', dataroot=TRAINVAL_PATH, verbose=verbose)
    elif dataset =="test":  
        nusc = NuScenes(version='v1.0-test', dataroot=TEST_PATH, verbose=verbose)

    elif dataset =="trainval_mini":
        nusc = NuScenes(version='v1.0-mini', dataroot=TRAINVAL_PATH_MINI, verbose=verbose)
    else:
        raise ValueError("Dataset must be either 'trainval' or 'test'")
    # token_list = get_sample_token_sequence(nusc)

    if verbose:
        print("Loading predictions...")
    res_boxes_full = load_prediction(result_path=detection_path, max_boxes_per_sample=300, box_cls=DetectionBox, verbose=True)
    res_boxes = res_boxes_full[0]
    pred = res_boxes.serialize()
    if verbose:
        print("Predictions succesfully loaded")

    if eval_split != 'test':
        if verbose:
            print("Loading ground truth...")
        gt_boxes = load_gt(nusc, eval_split = eval_split, box_cls=DetectionBox, verbose=True)
        gt = gt_boxes.serialize() 
        if verbose:
            print("Ground truth succesfully loaded")
    else:
        gt = None
    return nusc, gt, pred
     
        
def convert_history_to_dict(scene, history):
    temp = {}
    for i, token in enumerate(scene):
        tracks = []
        for track in history[i]:
            tracks.append(format_to_nuscene(token, track))
        temp[token] = tracks
    return temp


def save_to_json(results, str):
    print(f"Saving history to {str}")
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
        
    big_dic = {}
    big_dic['meta'] = {'use_camera': True, 'use_lidar': False, 'use_radar': False, 'use_map': False, 'use_external': False}
    big_dic['results'] = results

    # save history to json    
    with open(str, 'w') as fp:
        json.dump(big_dic, fp, cls=NumpyEncoder)
    
        
def ByteTrackv2(sample_tokens, pred, track_index, confidence_threshold, hung_thresh, verbose = False):  
    Covariance = covariance.Covariance()
    global tracks #将tracks变为全局更新
    tracks = [] 
    global history
    history = []

    if verbose:
        print('Start tracking...')
    for i, token in enumerate(sample_tokens): #每个scene的所有token
        if verbose:
            print("Frame", i)
        if i == 0:
            tracks = [] 
        Dhigh = []
        Dlow = []
        print("Token:", token)
        # detections = pred["e93e98b63d3b40209056d129dc53ceee"]
        # print("Detections:", detections)
        detections = pred[token] #获取当前token的所有detection
        for detection in detections:
            if detection["detection_name"] not in NUSCENES_TRACKING_NAMES:
                continue
            if detection["detection_score"] > confidence_threshold:
                Dhigh.append(detection)
            else:
                Dlow.append(detection)


        for track in tracks:
            track['state_predict'] = track['kalman'].predict()
            track['Found'] = False
        
        # 首次关联
        unmatched_detections_1 = association(Dhigh, tracks, hung_thresh=hung_thresh)
        
        # 二次关联
        _ = association(Dlow, tracks, hung_thresh=hung_thresh) 
        
        # 6帧未找到的track删除
        for track in tracks:
            if track['nb_lost_frame'] > 6: 
                if verbose : print("track deleted")
                tracks.remove(track)
            elif track['Found'] == False:
                # 如果没有找到，更新track的state
                track['state'][:-3] = track['state_predict'] 
                track['nb_lost_frame'] += 1

        # 创建新的track
        for d in unmatched_detections_1:
            track_index += 1
            tracking_name = d['detection_name']
            state = format_det_state(d)
            tracks.append({'state' : state, 
                        'tracking_name' : tracking_name,  
                        'track_id' : track_index,
                        'kalman' : KalmanFilter(state, DT, tracking_name, Covariance),
                        'state_predict': None,
                        'nb_lost_frame' : 0,
                        'Found' : False,
                        'velocity': d["velocity"],
                        'tracking_score': d["detection_score"]}) 
        tracks_copy = copy.deepcopy(tracks) #需要深拷贝，否则只是引用
        history.append(tracks_copy)
    if verbose: print("Tracking for this sequence done")

    for key in history:
        for track in key:
            del track['kalman']
    return history, track_index


def init(dataset, detection_path, eval_split, output_name, confidence_threshold=0.4, hungarian_threshold=0.6):
    print("Loading data")#nuScenes数据集处理部分
    nusc, gt, pred = load_data(dataset, detection_path, eval_split)
    # print("Keys in pred:", list(pred.keys()))
    val_token_list = []
    results = {} 
    track_index = 0
    token_list = get_sample_token_sequence(nusc)
    print("Prediction keys:", list(pred.keys()))
    # print("Token list:", token_list)
    for scene_tokens in token_list:
        # print("Checking scene_tokens:", scene_tokens)
        for tokens in scene_tokens:
            if not isinstance(tokens, list):#将tokens转换为list（commonelems参数包含两个list）
                tokens = [tokens]
            common_elements = commonelems(tokens, list(pred.keys()))
            if common_elements:
                print("Match found:", tokens)
                val_token_list.append(scene_tokens)
                break
            else:
                # print("No match found for:", tokens)
                pass
    #查找到的预测结果中包含的的token
    print("Val token list:", val_token_list)
    for scene_index in range(len(val_token_list)):
        scene = val_token_list[scene_index]
        print("scene:", scene)
        
        history, track_index = ByteTrackv2(scene, pred ,track_index, confidence_threshold, hungarian_threshold, verbose = True)#ByteTrackv2论文核心算法
        results = {**results, **convert_history_to_dict(scene, history)}
    save_to_json(results, output_name+".json")