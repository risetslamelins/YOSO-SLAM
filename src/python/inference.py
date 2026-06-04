# Modified by Romi Pamungkas from: https://github.com/iit-DLSLab/Panoptic-SLAM/blob/main/src/python/panoptic_python.py
# Modify Panoptic_Model in config/model_config.yaml to change models.

import numpy as np
import cv2
import yaml
from easydict import EasyDict
import argparse

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from predictor import VisualizationDemo

from meta_arch.yoso.config import add_yoso_config
from meta_arch.mask2former.config import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config

PANOPTIC_MODELS = ("YOSO", "Mask2Former", "PanopticFPN")


class PanopticSegmenter:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            yaml_content = file.readlines()

        # parse yaml content without opencv tag to prevent ScannerError
        if yaml_content and yaml_content[0].startswith("%YAML:1.0"):
            yaml_content = yaml_content[1:]
        slam_config = yaml.safe_load( "".join(yaml_content))
        
        if 'Panoptic_Model' not in slam_config:
            raise ValueError("'Panoptic_Model' key is missing from SLAM_Config.yaml file!")

        self.model_type = slam_config['Panoptic_Model']
        print(f"--- [INFO] Initializing PanopticSegmenter with model: {self.model_type} ---")

        if self.model_type not in PANOPTIC_MODELS:
            raise ValueError(
                f"Unsupported '{self.model_type}' as a panoptic model!"
                f"Supported models are: {PANOPTIC_MODELS}"
            )

        model_cfg = slam_config[self.model_type]

        args = self.setup_args(model_cfg)
        cfg = self.setup_cfg(args, self.model_type)
        self.demo = VisualizationDemo(cfg)

        self.output_img = None
        self.union_instance_mask = None
        self.image = None
    
    def setup_args(self, model_cfg):
        args_cfg = EasyDict()
        args_cfg.config_file = model_cfg['CONFIG']
        args_cfg.opts = ["MODEL.WEIGHTS", model_cfg['MODEL_PATH']]

        return args_cfg

    def setup_cfg(self, args, model_type):
        cfg = get_cfg()

        if model_type == "YOSO":
            add_yoso_config(cfg)
        elif model_type == "Mask2Former":
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
  
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)

        if model_type == "PanopticFPN":
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        
        cfg.freeze()
        return cfg
    
    def run_inference(self, img):
        self.image = img
        results = []
        self.all_masks = []
        
        predictions, visualized_output = self.demo.run_on_image(img)
        self.output_img = visualized_output.get_image()[:, :, ::-1]
        
        if self.model_type in ["YOSO", "Mask2Former"]:
            results = self._one_stage_panoptic(predictions)
        else: # PanopticFPN
            results = self._two_stage_panoptic(predictions)
            
        return results

    def _one_stage_panoptic(self, predictions):
        panoptic_mask_map, segments_info = predictions['panoptic_seg']
        panoptic_mask_map_np = panoptic_mask_map.cpu().numpy()

        results = []
        self.all_masks = []
        self.union_instance_mask = np.zeros(panoptic_mask_map_np.shape, dtype=np.uint8)

        for segment in segments_info:
            tmp = []

            binary_mask = (panoptic_mask_map_np == segment['id'])
            pred_mask = (binary_mask*255).astype('uint8')  

            # print(segment.items())

            tmp.append(segment['id'])
            tmp.append(segment['isthing'])
            tmp.append(segment.get('score', 0.0))
            tmp.append(segment['category_id'])
            tmp.append(0) #placeholder for instanceId
            tmp.append(segment['area'])
            
            if segment['isthing']:
                self.union_instance_mask[binary_mask] = 255
            
            tmp.append(self._binary_mask_2_bytearray(pred_mask)) # mask
            
            # bounding box
            if segment['isthing'] and segment['area'] > 0:
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                if rows.any() and cols.any():
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]
                    bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
                else:
                    bbox = [0.0, 0.0, 1.0, 1.0] # Placeholder for empty mask
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]  # Placeholder for stuff
            tmp.append(bbox)

            results.append(tmp)
            self.all_masks.append(pred_mask)
            
        return results 

    def _two_stage_panoptic(self, predictions):
        semantic_mask = predictions['sem_seg'].argmax(0).cpu()
        panoptic_result =predictions['panoptic_seg'][1]

        results = []
        self.all_masks = []

        instances = predictions['instances']
        fields = instances.get_fields() 
        instance_mask = instances.get('pred_masks').cpu().numpy() # collect all instance pred masks
        bboxes =  fields['pred_boxes'].tensor.cpu().numpy() # collect all instances bounding boxes
               
        #fill results with "Things" 
        #[id,isThing,score,category_id,instance_id,area,mask,bbox]
        num_instances = len(instance_mask)
        semantic_labels = [] # store semantic label 

        for info,pred_mask,boxes in zip(panoptic_result,instance_mask,bboxes):
            tmp=[]
            
            info_items = list(info.items()) # get each item from dict panoptic_result and convert into a list
            # print(info_items)

            pred_mask = np.array(pred_mask*255).astype('uint8')         

            #if isThing is False
            if(info_items[1][1] is False): continue # ignore Stuff objects 
            
            tmp.append(info_items[0][1]) # id
            tmp.append(info_items[1][1]) # isThing
            tmp.append(info_items[2][1]) #score
            tmp.append(info_items[3][1]) #category_id
            tmp.append(info_items[4][1]) #instance_id
            tmp.append(info_items[5][1]) #area
            tmp.append(self._binary_mask_2_bytearray(pred_mask)) # numpy mask
            tmp.append(list(boxes)) # numpy bbox
            results.append(tmp)
            self.all_masks.append(pred_mask)
     
        #fill results with "stuff" 
        semantic_results = panoptic_result[num_instances:]
        
        for info in semantic_results:
            info_items =list(info.items())
            semantic_labels.append(info_items[2][1])
   
        for x in np.unique(semantic_mask):
            if(x==0): 
                binary_mask = np.array(semantic_mask)==x
                self.union_instance_mask = binary_mask
            else:
                if(x in semantic_labels):
                    idx = semantic_labels.index(x)
                    binary_mask = np.array(semantic_mask)==x
                    semantic_result = semantic_results[idx]

                    info_items =list(semantic_result.items())
                    tmp = []

                    binary_mask = np.array(binary_mask*255).astype('uint8')

                    tmp.append(info_items[0][1]) # id
                    tmp.append(info_items[1][1]) # isThing
                    tmp.append(0.0) #score
                    tmp.append(info_items[2][1]) #category_id
                    tmp.append(0) #instance_id
                    tmp.append(info_items[3][1]) #area
                    tmp.append(self._binary_mask_2_bytearray(binary_mask)) # numpy mask
                    tmp.append([0,0,1,1]) # numpy bbox
                    results.append(tmp)
                    self.all_masks.append(binary_mask)
        
        return results

    def _binary_mask_2_bytearray(self, binary_mask):
        dst = cv2.merge((binary_mask,binary_mask,binary_mask))
        return bytearray(dst)
    
    # getter methods
    def get_output_img(self):
        return bytearray(self.output_img)

    def get_all_instance_mask(self):
        temp = np.zeros_like(self.image)
        v = Visualizer(temp, None, scale=1,instance_mode=2)   
        mask = v.draw_binary_mask(self.union_instance_mask,color='white',alpha=1)
        return bytearray(mask.get_image())

    def get_all_masks(self, width=640, height=480):
        masks = np.zeros([height,width,3], dtype=np.uint8)
        for mask in self.all_masks:
            dst = cv2.merge((mask,mask,mask))
            masks = cv2.addWeighted(masks,1,dst,1,0)
        return bytearray(masks)


if __name__=="__main__": 
    print("== Debug Panoptic Model Inference ==")

    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to config file")
    parser.add_argument("--img",type=str, required=True, help="image input path")  
    args=parser.parse_args()

    net = PanopticSegmenter(args.config)
    img = cv2.imread(args.img) 
    results = net.run_inference(img)

    cv2.imshow("Vis Result", net.output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()