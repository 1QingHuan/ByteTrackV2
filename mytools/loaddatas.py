from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt


TRAINVAL_PATH              = "data/trainval/"
TRAINVAL_PATH_MINI         = "data/trainval_mini/"
TEST_PATH                  = "data/testval/"

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
  ]

DT = 0.5

# import os
# import json
# from typing import List, Tuple, Dict

# class DetectionBox:
#     def __init__(self,name, score, bbox):
#         self.name = name
#         self.score = score
        

#     def serialize(self):
#         return {
#             'name': self.name,
#             'score': self.score
#         }
    
#     @classmethod
#     def deserialize(cls, data):
#         return cls(name=data['class'], score=data['confidence'])
    
# class EvalBoxes:
#     def __init__(self):
#         self.boxes = {}
#         self.sample_tokens = []

#     @classmethod
#     def deserialize(cls, data: List[Dict], box_cls):
#         eval_boxes = cls()
#         for entry in data:
#             class_name = entry['class']
#             confidence = entry['confidence']
#             bbox = entry['bbox']
#             sample_token = entry['sample_token']
            
#         return eval_boxes



# def load_prediction(result_path, max_boxes_per_sample, verbose = False):
#     # Load from file
#     with open(result_path) as f:
#         data = json.load(f)

#     # # Deserialize results
#     all_results = DetectionBox.deserialize(data)
#     if verbose:
#         print("Loaded results from {}. Found detections for {} samples."
#               .format(result_path, len(all_results.sample_tokens)))

    # # Check that each sample has no more than x predicted boxes.
    # for sample_token in all_results.sample_tokens:
    #     num_boxes = len(all_results.boxes[sample_token])
    #     if num_boxes > max_boxes_per_sample:
    #         raise ValueError("Error: Found {} boxes for sample {}. "
    #                          "Max boxes allowed is {}. "
    #                          "Please filter your results and try again."
    #                          .format(num_boxes, sample_token, max_boxes_per_sample))

    # return all_results


# def load_predictions_from_folder(folder_path, max_boxes_per_sample, box_cls, verbose = False):
#     all_results_list = []

#     # 排序文件夹中的json文件
#     json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')],
#                         key=lambda x: int(x.split('_')[1].split('.')[0]))

#     # 读取每个json文件
#     for file_name in json_files:
#         #获取file_name中的帧号，用于和all_results一起保存
#         frame_number = int(file_name.split('_')[1].split('.')[0])
#         result_path = os.path.join(folder_path, file_name)
#         all_results= load_prediction(result_path, max_boxes_per_sample, box_cls, verbose)
#         all_results_list.append((frame_number,all_results))
    
#     return all_results_list

def load_data(dataset=None, detection_path=None, eval_split=None, verbose=True):
    '''
    Dataset can be either "trainval" or "test", detection_path is the path to the detection folder.
    eval_split is the split of the dataset to evaluate on, can be either "train", "val" or "test".
    '''
    if dataset =="trainval":
        nusc = NuScenes(version='v1.0-trainval', dataroot=TRAINVAL_PATH, verbose=verbose)
    elif dataset =="test":  
        nusc = NuScenes(version='v1.0-test', dataroot=TEST_PATH, verbose=verbose)
    elif dataset =="trainval_mini":
        nusc = NuScenes(version='v1.0-mini', dataroot=TRAINVAL_PATH_MINI, verbose=verbose)
    else:
        print("数据集必须是'trainval'或'test'")

    if verbose:
        print("加载预测结果")
    res_boxes_full = load_prediction(result_path=detection_path, max_boxes_per_sample=300, box_cls=DetectionBox, verbose=True)
    res_boxes = res_boxes_full[0]
    pred = res_boxes.serialize()#以字典形式返回
    if verbose:
        print("预测结果加载成功")

    if eval_split != 'test':
        if verbose:
            print("加载正确标签")
        gt_boxes = load_gt(nusc, eval_split = eval_split, box_cls=DetectionBox, verbose=True)
        gt = gt_boxes.serialize() 
        if verbose:
            print("正确标签加载成功")
    else:
        gt = None
    return nusc, gt, pred


