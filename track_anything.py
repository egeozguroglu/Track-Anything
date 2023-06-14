import PIL
import time
import cv2
import time
import psutil
import os
import torchvision
import torch 
from tqdm import tqdm

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse

class TrackingAnything():
    def __init__(self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)
        self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device) 
    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

    # template_mask: masks 'to track'
    def generator(self, images: list, template_mask:np.ndarray):  
        masks = []
        logits = []
        overlays = []
        ious = []
        for i in tqdm(range(len(images)), desc="Tracking the provided template mask through images"):
            if i == 0:           
                mask, logit, inpainted_overlay = self.xmem.track(images[i], template_mask)
            else:
                mask, logit, inpainted_overlay = self.xmem.track(images[i])

            pred_iou = iou(mask, template_mask)
            # ious.append(pred_iou)
            # print(torch.unique(logit[0]))
            # print(torch.unique(logit[1]))
            # print(pred_iou)
            # print(torch.unique(mask))
            masks.append(mask)
            logits.append(logit.cpu().numpy())
            overlays.append(inpainted_overlay)
            ious.append(pred_iou)
           
        return masks, logits, overlays, ious
    
def iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    return np.sum(intersection) / np.sum(union)

def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def get_frames_from_video(video_input):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = list()
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1])
    print("Shape of images in video: %s" % image_size)
    return frames
        
def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:2")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=True)
    args = parser.parse_args()
    if args.debug:
        print(args)
    return args 

if __name__ == "__main__":
    args = parse_augment()
    
    template_mask = np.load("/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/result/mask/test_video/00000.npy")
    images = get_frames_from_video("/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/test_sample/test_video.mp4")

    trackany = TrackingAnything(sam_checkpoint= '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/sam_vit_h_4b8939.pth',
                                xmem_checkpoint= '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/XMem-s012.pth', 
                                e2fgvi_checkpoint = '/proj/vondrick4/ege/workspace/future-objects/labeling/Track-Anything/checkpoints/E2FGVI-HQ-CVPR22.pth',
                                args = args)
    
    masks, logits, overlay_images= trackany.generator(images, template_mask)
    video_output = generate_video_from_frames(overlay_images, output_path="./result/track/{}".format("test_output.mp4"), fps=24)

        
        
