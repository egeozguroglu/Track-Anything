import PIL
import time
import cv2
import time
import psutil
import os
import json
from tqdm import tqdm
from PIL import Image

from track_anything import TrackingAnything, get_frames_from_video, generate_video_from_frames
from utils import overlay_semantic_mask, davis_palette # refactor properly (originally in future-objects/datasets)
import numpy as np
import argparse
import glob

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:2")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--mask_save', default=True)
    args = parser.parse_args()
    return args 

def get_frame_paths_from_index(frame_idx, video_dir):
    frame_idx_str = str(frame_idx).zfill(10)
    frame_dir = video_dir + frame_idx_str + "/"

    # TODO: frames without VISOR annotations! i.e. frames without directories (e.g. frame_010200020.jpg) that don't have annotations.
    #
    # This sparsity is exactly why we're doing backward tracking! Need to backward track through them! 
    # 
    # We can change processing to make directories for such frames as well, allowing us to 
    # backtrack active objects to those frames as well (instead of just frames with directories).
    if not os.path.exists(frame_dir):
        rgb_path = video_dir + "/frame_{}.jpg".format(frame_idx_str)
        return frame_dir, rgb_path, None, None 

    rgb_path = frame_dir + frame_idx_str + "_rgb.jpg"
    mask_path = frame_dir + frame_idx_str + "_mask.png"
    hand_obj_path = frame_dir + frame_idx_str + "_handobject.json"
    return frame_dir, rgb_path, mask_path, hand_obj_path

def get_video_length(video_dir):
    # frame_dirs = glob.glob(os.path.join(video_dir, '*.jpg))
    frame_paths = [x for x in os.listdir(video_dir) if not x.startswith('.')]
    assert len(frame_paths) > 0, "No frames found in video directory: {}".format(video_dir)
    return len(frame_paths)

def get_frames_from_path(video_dir):
    frames = list()

    for frame_idx in tqdm(range(1, 3585+1)): # for testing
    # for frame_idx in tqdm(range(1, get_video_length(video_dir)+1)):
        frame_dir, rgb_path, mask_path, hand_obj_path = get_frame_paths_from_index(frame_idx, video_dir)

        if not os.path.exists(frame_dir):
            # check if rgb exists:
            if not os.path.isfile(rgb_path):
                print(rgb_path)
                continue
            
        frame_rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    return frames


def generate_masks_for_image(
        rgb,
        masks_info, 
        input_resolution=(480, 854), 
        output_resolution=(480, 854)):

    mask = np.zeros([input_resolution[0], input_resolution[1]], dtype=np.uint8)
    print(len(masks_info))

    for entity in masks_info:

        # testing with active object fridge, class_id = 12
        if entity["class_id"] != 41:
            continue

        ################

        object_annotations = entity["segments"]
        polygons = []
        polygons.append(object_annotations)
        ps = []
        for polygon in polygons:
            for poly in polygon:
                if poly == []:
                    poly = [[0.0, 0.0]]
                ps.append(np.array(poly, dtype=np.int32))
        cv2.fillPoly(mask, pts=ps, color=(entity["class_id"], entity["class_id"], entity["class_id"]))

    mask = cv2.resize(mask, (output_resolution[1], output_resolution[0]), interpolation=cv2.INTER_NEAREST)
    overlay = overlay_semantic_mask(rgb, mask, colors = davis_palette)
    return mask, overlay



# 1. Which occurence of the active object we start from (where we start backtracking): where it becomes 'active' wrt VISOR
# 2. How far we track an active object backward
# 3. Tracking multiple active objects together or seperately?  
##### 
def find_active_object(video_path):
    active_object_id = None # selected active object

    # start from the end of the video
    for frame_idx in tqdm(reversed(range(1, get_video_length(video_path) + 1))):
        frame_dir, rgb_path, mask_path, annot_json = get_frame_paths_from_index(frame_idx, video_path)
    
        # find frame with annotation containing an active object (not hand)
        if not os.path.exists(frame_dir):
            continue

        with open(annot_json) as f:
            frame_data = json.load(f)

            for entity in frame_data["annotations"]:
                # skip left hand and right hand
                if (entity["class_id"] == 301) or (entity["class_id"] == 300):
                    continue
                
                active_object_id = entity["class_id"]
                

    # find where VISOR annotations for that active object ends (or starts, in forward terms)

    # from the end of VISOR annotations (where it first becomes active), start tracking object backwards
    
    # backward track until object goes out of view
    pass




def objectjson_to_mask(rgb_path, json_path):

    rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    print(rgb_img.shape)

    with open(json_path) as f:
        frame_data = json.load(f)
        mask, overlay = generate_masks_for_image(rgb_img, 
                                                 frame_data["annotations"], 
                                                 output_resolution=rgb_img.shape)
    # overlay_img = Image.fromarray(overlay)
    # overlay_img.save("overlay_mask.png")
    return mask


def backward_tracking(tracker, active_obj_mask, video_frames):
    track_mask(tracker, active_obj_mask, video_frames[::-1])


def track_mask(tracker, mask_to_be_tracked, video_frames):

    masks, logits, overlay_images = tracker.generator(video_frames, mask_to_be_tracked)

    video_output = generate_video_from_frames(overlay_images, output_path="./result/track/{}".format("backward_carrot_3585to1000.mp4"), fps=24)


def main():
    args = parse_augment()
    
    mask_to_be_tracked = objectjson_to_mask("/proj/vondrick4/ege/datasets/future-objects/EPIC-KITCHENS/P01/rgb_frames/P01_01/0000003585/0000003585_rgb.jpg",
        "/proj/vondrick4/ege/datasets/future-objects/EPIC-KITCHENS/P01/rgb_frames/P01_01/0000003585/0000003585_handobject.json")
    print(np.unique(mask_to_be_tracked))
    
    
    video_frames = get_frames_from_path("/proj/vondrick4/ege/datasets/future-objects/EPIC-KITCHENS/P01/rgb_frames/P01_01/")
    # images = get_frames_from_video("/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/test_sample/test_video.mp4")
    print(len(video_frames))

    tracker = TrackingAnything(sam_checkpoint= '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/sam_vit_h_4b8939.pth',
                                xmem_checkpoint= '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/XMem-s012.pth', 
                                e2fgvi_checkpoint = '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/E2FGVI-HQ-CVPR22.pth',
                                args = args)
    
    backward_tracking(tracker, mask_to_be_tracked, video_frames)
    


if __name__ == "__main__":
    main()