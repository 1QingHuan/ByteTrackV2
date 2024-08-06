import mytools.ByteTrackv2 as ByteTrackv2

import mytools.visualisation as visualisation
import matplotlib.pyplot as plt

DETECTION_PATH = "results_nusc.json"#for testing

detection_path = DETECTION_PATH

nusc, gt, pred = ByteTrackv2.load_data(dataset = "trainval", detection_path = DETECTION_PATH, eval_split = "val", verbose = False)

token_list = ByteTrackv2.get_sample_token_sequence(nusc)
val_token_list = ByteTrackv2.get_common_tokens(token_list, pred)
print(f"there are {len(val_token_list)} samples in the validation set")

scene_index = 0 # has to be between 0 and len(val_token_list)-1
history, _ = ByteTrackv2.ByteTrackv2(val_token_list[scene_index], pred, 0, confidence_threshold=0.4, hung_thresh=0.6, verbose = False) # run bytettrack on the desired scene

tokens = val_token_list[scene_index]
visualisation.create_gif(tokens, nusc, gt, pred, history, 0.6, plot_gt=True, plot_pred=False, plot_track=True)