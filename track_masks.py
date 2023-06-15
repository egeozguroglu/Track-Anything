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
    rgb_path = frame_dir + frame_idx_str + "_rgb.jpg"
    mask_path = frame_dir + frame_idx_str + "_mask.png"
    hand_obj_path = frame_dir + frame_idx_str + "_handobject.json"
    return frame_dir, rgb_path, mask_path, hand_obj_path

def get_video_length(video_dir):
    frame_paths = [x for x in os.listdir(video_dir) if not x.startswith('.')]
    assert len(frame_paths) > 0, "No frames found in video directory: {}".format(video_dir)
    return len(frame_paths)

def get_frames_from_path(video_dir):
    frames = list()

    for frame_idx in tqdm(range(75100, 75571+1)): # for testing 
    # for frame_idx in tqdm(range(1, get_video_length(video_dir)+1)):
        frame_dir, rgb_path, mask_path, hand_obj_path = get_frame_paths_from_index(frame_idx, video_dir)

        if not os.path.exists(frame_dir):
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

    for entity in masks_info:
        print(entity["class_id"])

        # testing with active object class id
        if entity["class_id"] != 29:
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

    print(np.unique(mask))
    mask = cv2.resize(mask, (output_resolution[1], output_resolution[0]), interpolation=cv2.INTER_NEAREST)
    overlay = overlay_semantic_mask(rgb, mask, colors = davis_palette)
    return mask, overlay


''' 
Provided a video path and the class id of an active object that's known to be present in video, 
backtrack the object throughout the video. 
'''
def backtrack_used_object(video_path, active_object_classid, tracker):

    current_frame_idx = get_video_length(video_path) # start from the end of the video

    # continue backtracking until reaching the start of the video
    while current_frame_idx != 0:
        
        ## (1) Find where hand contact (usage) occurs with specified active object.
        used_index = None
        for frame_idx in reversed(range(1, current_frame_idx + 1)):
            frame_dir, rgb_path, mask_path, json_path = get_frame_paths_from_index(frame_idx, video_path)
            if not os.path.isfile(json_path):
                continue

            frame_data = read_annotation_json(json_path)
            objects_in_handcontact = frame_data["hand_object"].values()
            current_frame_idx -= 1
            
            if active_object_classid in objects_in_handcontact: 
                used_index = frame_idx
                print("Frame: %d, In Hand Contact" % frame_idx)
                break
        
        if used_index:
            ## (2) Find where VISOR active annotations end (in backward terms) to determine backtracking start index.
            backtracking_start_idx = used_index # if no active annotations after hand contact

            for frame_idx in reversed(range(1, used_index + 1)):
                frame_dir, rgb_path, mask_path, json_path = get_frame_paths_from_index(frame_idx, video_path)
                if not os.path.isfile(json_path):
                    continue

                frame_data = read_annotation_json(json_path)
                objects_active = [obj["class_id"] for obj in frame_data["annotations"]]
                current_frame_idx -= 1

                if active_object_classid in objects_active: 
                    backtracking_start_idx = frame_idx
                    backtracking_mask = objectjson_to_mask
                else:
                    break

            ## (3) From the end of VISOR annotations (where it first becomes active), backtrack the object. 
            btr_frame_dir, btr_rgb_path, btr_mask_path, btr_json_path = get_frame_paths_from_index(frame_idx, video_path)
            backtracking_mask = objectjson_to_mask(btr_rgb_path, btr_json_path)
            
            all_frames = get_frames_from_path(video_path)[::-1]
            frames_to_be_tracked = all_frames[backtracking_start_idx: backtracking_start_idx + 1000] # can go until the end if stable

            track_mask(tracker, backtracking_mask, frames_to_be_tracked)

def backward_tracking(tracker, active_obj_mask, video_frames):
    track_mask(tracker, active_obj_mask, video_frames[::-1])

def track_mask(tracker, mask_to_be_tracked, video_frames):

    masks, logits, overlay_images = tracker.generator(video_frames, mask_to_be_tracked)
    # logit: 2 (background, foreground) H, W

    counter = 0
    for frame_idx in reversed(range(2800, 2064+1)):
        print("Frame: %d" % frame_idx)
        bg_probs = logits[counter][0]
        print(np.mean(bg_probs.squeeze()))

        fg_probs = logits[counter][0]
        print(np.mean(fg_probs.squeeze()))
        print(np.mean(fg_probs.squeeze()[masks[counter] == 29]))
        print("------------------")
        counter += 1


    video_output = generate_video_from_frames(overlay_images, output_path="./result/track/{}".format("backward_pot.mp4"), fps=24)


'''
Retrieves class id's of all unique active objects at the specified video path.
'''
def find_all_activeobjects_in_video(video_path):
    active_objects = list()
    for frame_idx in tqdm(range(1, get_video_length(video_path) + 1)):
        frame_dir, rgb_path, mask_path, json_path = get_frame_paths_from_index(frame_idx, video_path)

        # if no VISOR annotation for frame, skip. 
        if not os.path.isfile(json_path):
            continue

        frame_data = read_annotation_json(json_path)
        for entity in frame_data["annotations"]:
            class_id = entity["class_id"]
            class_name = entity["name"]
            # skip left and right hands
            if (class_id == 301) or (class_id == 300):
                continue
            if (class_id, class_name) not in active_objects:
                active_objects.append((class_id, class_name))
        # print(active_objects)
    return active_objects


def read_annotation_json(json_annotation_path):
    with open(json_annotation_path, 'r') as f:
        annotation_data = json.load(f)
    return annotation_data


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



def main():
    video_path = "/proj/vondrick4/ege/datasets/future-objects/EPIC-KITCHENS/P01/rgb_frames/P01_05/"

    args = parse_augment()
    
    mask_to_be_tracked = objectjson_to_mask("/local/crv/neilnie/datasets/EPIC-KITCHENS/P01/rgb_frames/P01_05/0000003064/0000003064_rgb.jpg",
        "/local/crv/neilnie/datasets/EPIC-KITCHENS/P01/rgb_frames/P01_05/0000003064/0000003064_handobject.json")
    # print(np.unique(mask_to_be_tracked))
     
    video_frames = get_frames_from_path(video_path)
    print(len(video_frames))

    tracker = TrackingAnything(sam_checkpoint= '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/sam_vit_h_4b8939.pth',
                                xmem_checkpoint= '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/XMem-s012.pth', 
                                e2fgvi_checkpoint = '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/E2FGVI-HQ-CVPR22.pth',
                                args = args)
    
    backward_tracking(tracker, mask_to_be_tracked, video_frames)
    '''
    # active_objects = find_all_activeobjects_in_video(video_path)
    # print(active_objects)
    # active_object_ids = [x[0] for x in active_objects]
    # print("Backtracking " + active_objects[0][1])
    backtrack_active_object_in_video(video_path, 24)
    '''


if __name__ == "__main__":
    main()