import mytools.ByteTrackv2 as ByteTrackv2

# DETECTION_PATH ="runs/detect/exp3"#for mydateset
# DETECTION_PATH = "Detect.json"#for testing
DETECTION_PATH = "results_nusc.json"#for testing

detection_path = DETECTION_PATH
dataset='trainval'
eval_split="train"
# dataset='test'
# dataset='trainval_mini'
# eval_split="mini_train"

output_name="ByteTrackV2"

ByteTrackv2.init(dataset, detection_path, eval_split, output_name)
# ByteTrackV2.init(detection_path)