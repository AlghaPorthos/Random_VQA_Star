import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from PIL import Image, UnidentifiedImageError  
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from IPython.display import display
from PIL import Image
import base64
import random
import json
from torchvision import transforms
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time
import dashscope
from datetime import datetime
from openai import APIConnectionError
from tqdm import tqdm
from openai import OpenAI
import argparse

# HOME = os.getcwd()
# print(HOME)
model = None
filename = None
client = None
"""
Run with:
CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com \
  WEIGHTS_PATH = "/home/fujl/Grounding_Dino_Test/GroundingDINO/weights/groundingdino_swint_ogc.pth" \
  json_dir = "/home/fujl/Grounding_Dino_Test/MME-RealWorld_Part/MME_RealWorld_RS_and_AD.json" \
  CONFIG_PATH = "/home/fujl/Grounding_Dino_Test/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  HOME = "/home/fujl/Grounding_Dino_Test/MME-RealWorld_Part" \
  API_KEY = "sk-4fe420a269a14933b5cb36811878e4aa" \
  python run_without_position_prompt.py

"""



# WEIGHTS_PATH = "/home/fujl/Grounding_Dino_Test/GroundingDINO/weights/groundingdino_swint_ogc.pth"
# json_dir = "/home/fujl/Grounding_Dino_Test/MME-RealWorld_Part/MME_RealWorld_RS_and_AD.json"
# CONFIG_PATH = "/home/fujl/Grounding_Dino_Test/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# HOME = "/home/fujl/Grounding_Dino_Test/MME-RealWorld_Part"

# print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
# print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))



def position_cue_qwen(sentence, w = 1.00, h = 1.00):
    global client
    expected_positions = ["right edge", "left edge", "top edge", "bottom edge", "right", "left", "top", "bottom", "bottom right corner", "bottom left corner", "top left corner", "top right corner", "lower right corner", "lower left corner", "upper left corner", "upper right corner", "bottom right", "bottom left", "top left", "top right", "lower right", "lower left", "upper left", "upper right", "central", "middle", "middle right", "middle left", "upper middle", "lower middle", "top middle", "bottom middle", "full frame"]
    text_to_bbox = {
      "right edge":            ((3.00*w)/4.00, 0, w, h),
      "left edge":            (0, 0, w/4.00, h),
      "top edge":             (0, 0, w, (h*1.00)/4.00),
      "bottom edge":           (0, h*3.00/4.00, w, h),

      "right":		    ((2.00*w)/3.00, 0, w, h),
      "left":			   (0, 0, w/3.00, h),
      "top":			   (0, 0, w, (h*1.00)/3.00),
      "bottom":		    (0, h*2.00/3.00, w, h),

      "bottom right corner":  		 (w*2.00/3.00, h*2.00/3.00, w, h),
      "bottom left corner":   		 (0, h*2.00/3.00, w/3.00, h),
      "top left corner":      		 (0, 0, w/3.00, h/3.0),
      "top right corner":     		 (w*2.00/3.00, 0, w, h/3.00),
      "lower right corner":   		 (w*2.00/3.00, h*2.00/3.00, w, h),
      "lower left corner":    		 (0, h*2.00/3.00, w/3.00, h),
      "upper left corner":    	 (0, 0, w/3.00, h/3.00),
      "upper right corner":   		 (w*2.00/3.00, 0, w, h/3.00),

      "bottom right":   		 (w*2.00/3.00, h*2.00/3.00, w, h),
      "bottom left":    		 (0, h*2.00/3.00, w/3.00, h),
      "top left":       		 (0, 0, w/3.00, h/3.0),
      "top right":      		 (w*2.00/3.00, 0, w, h/3.00),
      "lower right":    		 (w*2.00/3.00, h*2.00/3.00, w, h),
      "lower left":     		 (0, h*2.00/3.00, w/3.00, h),
      "upper left":    		 (0, 0, w/3.00, h/3.00),
      "upper right":    		 (w*2.00/3.00, 0, w, h/3.00),

      "central":               (w/4.00, h/4.00,  (3.00*w)/4.00, (3.00*h)/4.00),
      "middle":                (w/4.00, h/4.00,  (3.00*w)/4.00, (3.00*h)/4.00),

      "middle right":          (w*2.00/3.00, h/4.00, w, (h*3.00)/4.00),
      "middle left":           (0, h/4.00, w/3.00, (h*3.00)/4.00),
      "upper middle":          (w/4.00, 0, (3.00*w)/4.00, h/3.00),
      "lower middle":          (w/4.00, h*2.00/3.00, (3.00*w)/4.00, h),

      "top middle":           (w/4.00, 0, (3.00*w)/4.00, h/3.00),
      "bottom middle":          (w/4.00, h*2.00/3.00, (3.00*w)/4.00, h),

      "full frame":            (0, 0, w, h),
    }


    # user_msg = "" + sentence + "Please give me the most related region according to the position cue the question asked. Please select from the following word WITHOUT any other word: right edge, left edge, top edge, bottom edge, right, left, top, bottom, bottom right corner, bottom left corner, top left corner, top right corner, lower right corner, lower left corner, upper left corner, upper right corner, bottom right, bottom left, top left, top right, lower right, lower left, upper left, upper right, central, middle, middle right, middle left, upper middle, lower middle, top middle, bottom middle, full frame. If there is none, use \'full frame\'"
    most_similar = "full frame"
    for i in range(1):
        # qwen_agent.reset()
        # response = qwen_agent.step(user_msg)

        try:
          completion = client.chat.completions.create(
              model="qwen-plus",
              messages=[
                  {'role': 'system', 'content': 'You will be given a question, Please give me the most related region according to the position cue the question asked. Please select from the following word WITHOUT any other word: right edge, left edge, top edge, bottom edge, right, left, top, bottom, bottom right corner, bottom left corner, top left corner, top right corner, lower right corner, lower left corner, upper left corner, upper right corner, bottom right, bottom left, top left, top right, lower right, lower left, upper left, upper right, central, middle, middle right, middle left, upper middle, lower middle, top middle, bottom middle, full frame. If there is none, use \'full frame\''},
                  {'role': 'user', 'content': sentence}],
              )
          response_content = completion.choices[0].message.content
          print(response_content)
        except APIConnectionError as e:
            print(f"Skipped with Connection Error:  {e}")
            return [0,0,w,h], "full frame"
        except Exception as e:
            print(f"Skipped with Other Error:  {e}")
            return [0,0,w,h], "full frame"

        # print(response.msgs[0].content)
        if response_content in expected_positions:
            most_similar = response_content
            break
        else:
            print("Response not in expected positions, retrying...")
            # pass
    # print(most_similar)
    range_of_text_to_bbox = text_to_bbox.get(most_similar)
    # print(range_of_text_to_bbox)
    final_bbox = range_of_text_to_bbox
    if w>=10 and h >= 10:
      final_bbox = [int(range_of_text_to_bbox[0]), int(range_of_text_to_bbox[1]), int(range_of_text_to_bbox[2]), int(range_of_text_to_bbox[3])]
    return final_bbox, most_similar


def remove_position_cue_qwen(sentence):
  global client
  for i in range(1):
    related_bbox, pos_cue = position_cue_qwen(sentence)
    if pos_cue == 'full frame':
      # print(sentence)
      return sentence
    else:
      print("Re-generating: " + str(i+1) + " of 10")
    # user_msg = "Here is a question: " + sentence + "This question has attention of " + pos_cue + " of the image. But acturally the image may be cropped in the process, so any position cue based on the whole image is invalid. Now please remove the position cue related to " + pos_cue + " based on the whole image (note that this is not paraphrasing), but keep OTHER information all the same."

    # completion = client.chat.completions.create(
    #     model="qwen-plus",
    #     messages=[
    #         {'role': 'system', 'content': "You will be given a question, This question has attention of " + pos_cue + " of the image. But acturally the image may be cropped in the process, so any position cue based on the whole image is invalid. Now please remove the position cue related to " + pos_cue + " based on the whole image (note that this is not paraphrasing), but keep OTHER information all the same."},
    #         {'role': 'user', 'content': sentence}],
    #     )
    # # 获取助手的回复内容
    # sentence = completion.choices[0].message.content

    try:
      completion = client.chat.completions.create(
          model="qwen-plus",
          messages=[
              {'role': 'system', 'content': "You will be given a question, This question has attention of " + pos_cue + " of the image. But acturally the image may be cropped in the process, so any position cue based on the whole image is invalid. Now please remove the position cue related to " + pos_cue + " based on the whole image (note that this is not paraphrasing), but keep OTHER information all the same."},
              {'role': 'user', 'content': sentence}],
          )
      # 获取助手的回复内容
      sentence = completion.choices[0].message.content
      print(sentence)
    except APIConnectionError as e:
        print(f"连接错误: {e}")
        # 可以选择重试或返回默认值
        return sentence
    except Exception as e:
        print(f"发生错误: {e}")
        return sentence
    print(sentence)
  return sentence


def calculate_iou(box1, box2):
    # box1 and box2 are both in xyxy format
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    intersection_width = max(0, x2_intersection - x1_intersection)
    intersection_height = max(0, y2_intersection - y1_intersection)
    intersection_area = intersection_width * intersection_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou
def get_sub_patches_by_random_cropping(current_patch_bbox, suffix, heatmap): # used as part of A*
  x1, y1, x2, y2 = current_patch_bbox
  x0, y0 = x1, y1
  w = x2 - x1
  h = y2 - y1
  bboxes = []
  for _ in range(1000):
    x1 = np.random.randint(0, 50) * 10 + 5
    y1 = np.random.randint(0, 50) * 10 + 5
    x2 = np.random.randint(0, 50) * 10 + 5
    y2 = np.random.randint(0, 50) * 10 + 5
    if x1 > x2:
      x1, x2 = x2, x1
    if y1 > y2:
      y1, y2 = y2, y1
    if abs((y2-y1) * (x2-x1)) > 0.1*(500*500) and abs((y2-y1) * (x2-x1)) <= 0.9*(500*500):
      current_heatmap = heatmap[y1:y2, x1:x2]
      x1 = x1/500.00
      y1 = y1/500.00
      x2 = x2/500.00
      y2 = y2/500.00
      extended = extend_bbox([x1, y1, x2, y2])
      x1, y1, x2, y2 = extended
      x1 = int(x1*w)
      y1 = int(y1*h)
      x2 = int(x2*w)
      y2 = int(y2*h)
      bboxes.append([[x1 + x0, y1 + y0, x2 + x0, y2 + y0], current_heatmap.mean() + (np.log10(y2-y1) + np.log10(x2-x1))/5])
  bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
  # print("--------Printing get_sub_patches_by_random_cropping --------")
  # for i in range(5):
  #   print(bboxes[i])
  # print("------------------------------------------------------------")
  crop_coords = []
  cnt_bbox = 0
  for i in range(len(bboxes)):
    flag = True
    for j in range(i):
      # print(i, j, calculate_iou(bboxes[i][0], bboxes[j][0]))
      if(calculate_iou(bboxes[i][0], bboxes[j][0]) > 0.8): # which means that the two boxes are too close
        flag = False
         
        
    if flag == True:
      crop_coords.append((suffix + str(cnt_bbox), bboxes[i][0]))
      cnt_bbox += 1
    if cnt_bbox == 6:
      break
  # print("--------Printing get_sub_patches_by_random_cropping --------")
  # for crop_coord in crop_coords:
  #   print(crop_coord)
  # print("------------------------------------------------------------")
  return crop_coords # for suffix, start, end in crop_coords:

def extend_bbox(bbox):
    x1, y1, x2, y2 = bbox  # in xyxy format, and all values are in range of (0,1)

    center_x = (x1 + x2) / 2.00
    center_y = (y1 + y2) / 2.00

    width = x2 - x1
    height = y2 - y1

    new_width = width * 2
    new_height = height * 2

    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    new_x1 = max(0.0, new_x1)
    new_y1 = max(0.0, new_y1)
    new_x2 = min(1.0, new_x2)
    new_y2 = min(1.0, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]


def get_judge_heatmap(image, text_prompt, bbox, boxes, logits, phrases):
  BOX_TRESHOLD = 0.25
  TEXT_TRESHOLD = 0.2
  image = image.crop(bbox)
  transform = T.Compose(
      [
          T.RandomResize([800], max_size=1333),
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ]
  )
  

  image_source = image.convert("RGB")
  image_np = np.asarray(image_source)
  image_transformed_np, _ = transform(image_source, None)

  heatmap = np.zeros((500, 500))
  np_width = 500
  np_height = 500
  for i in range(len(boxes)-1, -1, -1):  # in reversed order, so the bbox with larger index will be drawn on top
    # print(boxes[i])
    x_center, y_center, width, height = boxes[i].tolist()
    x_min = int((x_center - width / 2) * 500)
    y_min = int((y_center - height / 2) * 500)
    x_max = int((x_center + width / 2) * 500)
    y_max = int((y_center + height / 2) * 500)
    # print(x_min, y_min, x_max, y_max)
    bbox_i_01 = extend_bbox([x_min*1.00/np_width, y_min*1.00/np_height, x_max*1.00/np_width, y_max*1.00/np_height])
    x_min, y_min, x_max, y_max = bbox_i_01
    x_min = int(x_min * np_width)
    y_min = int(y_min * np_height)
    x_max = int(x_max * np_width)
    y_max = int(y_max * np_height)
    # bbox_xyxy = (x_min, y_min, x_max, y_max)
    # print("get_judge_heatmap: ",bbox_i_01, bbox_xyxy, ": ", logits[i], phrases[i])
    # print(bbox_xyxy, logits[i], phrases[i])
    heatmap[y_min:y_max,x_min:x_max] = float(logits[i])

  return heatmap


def get_judge_value_without_noun_list(image, text_prompt, bbox):
  global model
  image = image.crop(bbox)

  transform = T.Compose(
      [
          T.RandomResize([800], max_size=1333),
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ]
  )

  image_source = image.convert("RGB")
  image_np = np.asarray(image_source)
  image_transformed_np, _ = transform(image_source, None)

  BOX_TRESHOLD = 0.25
  TEXT_TRESHOLD = 0.2

  boxes, logits, phrases = predict(
      model=model,
      image=image_transformed_np,
      caption=text_prompt,
      box_threshold=BOX_TRESHOLD,
      text_threshold=TEXT_TRESHOLD
  )
  heatmap_returned = get_judge_heatmap(image, text_prompt, bbox, boxes, logits, phrases)
  return heatmap_returned, heatmap_returned.max()*10+heatmap_returned.mean() # + (np.log10(bbox[2] - bbox[0]) + np.log10(bbox[3] - bbox[1]))*0.50


class Prioritize: # class from penghao-wu/vstar
	def __init__(self, priority, item):
		self.priority = priority
		self.item = item

	def __eq__(self, other):
		return self.priority == other.priority

	def __lt__(self, other):
		return self.priority < other.priority


def remove_substrings(input_list):
    unique_list = []

    for i in range(len(input_list)):
        is_substring = False
        for j in range(len(input_list)):
            if input_list[j] in input_list[i] and not (input_list[i] == input_list[j]):
                is_substring = True
                break
        if not is_substring:
            unique_list.append(input_list[i])

    final_list = []
    for item in unique_list:
        if not any(item in existing for existing in final_list):
            final_list.append(item)

    return final_list


def Simple_VG(image_dir, question, min_res = 1500):
  print(image_dir)
  print(question)
  image = Image.open(image_dir)
  
  related_bbox, pos_cue = position_cue_qwen(question, image.width, image.height)
  related_bbox_2 = related_bbox

  question_without_pos = remove_position_cue_qwen(question)
  print(f"question_without_pos: {question_without_pos}")
  queue = PriorityQueue()
  transform = T.Compose(
    [
      T.RandomResize([800], max_size=1333),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
  )

  # image_source = image.convert("RGB")
  # image_np = np.asarray(image_source)
  # image_transformed_np, _ = transform(image_source, None)

  # _, _, all_phrases = predict(
  #     model=model,
  #     image=image_transformed_np,
  #     caption=question,
  #     box_threshold=0.1,
  #     text_threshold=0.1
  # )
  # all_phrases = remove_substrings(all_phrases)
  # unsorted_noun_list = []

  # for noun in all_phrases:
  #   if noun in unsorted_noun_list:
  #     pass
  #   else:
  #     unsorted_noun_list.append(noun)

  # most_related_noun = find_most_related_obj(question, unsorted_noun_list)

  # noun_list = [most_related_noun]

  # for noun in all_phrases:
  #   if noun in noun_list:
  #     pass
  #   else:
  #     noun_list.append(noun)

  # print("-------------------------")
  # print("noun_list: ", noun_list)
  # print("most_related_noun: ", most_related_noun)
  # print("-------------------------")

  init_patch = dict()
  init_patch['bbox'] = related_bbox
  init_patch['suffix'] = "_"
  # init_patch['priority_score'] = 1
  # init_patch['priority_score'] = get_judge_value(image, question, init_patch['bbox'], noun_list)
  init_patch['heatmap'], init_patch['priority_score'] = get_judge_value_without_noun_list(image, question, init_patch['bbox'])


  cnt_item_queue = 0
  queue.put(Prioritize(-init_patch['priority_score'], init_patch))
  # print("queue.empty():", queue.empty())
  max_patch = init_patch
  history_bbox = []
  while(not queue.empty()): # the main loop of A_star searching
    patch_chosen = queue.get().item
    cnt_item_queue += 1
    current_bbox = patch_chosen['bbox']
    current_suffix = patch_chosen['suffix']
    current_heatmap = patch_chosen['heatmap']
    current_priority = patch_chosen['priority_score']
    if current_bbox in history_bbox:
      print(f"SKIPPED :: current_suffix: {current_suffix}, ccurrent_bbox: {current_bbox}")
      continue
    else:
      history_bbox.append(current_bbox)

    if current_priority > max_patch['priority_score']:
      max_patch = patch_chosen

    print(f"current_suffix: {current_suffix}, current priority: {current_priority}, max pri score: {max_patch['priority_score']}, ccurrent_bbox: {current_bbox}")

    if 2 * (current_priority) < max_patch['priority_score'] or cnt_item_queue > 20:
      break
    # crop_coords = get_sub_patches(current_bbox, current_suffix)
    crop_coords = get_sub_patches_by_random_cropping(current_bbox, current_suffix, current_heatmap)

    # print(crop_coords)

    if max(current_bbox[2] - current_bbox[0], current_bbox[3] - current_bbox[1]) >= min_res:
      for clip_suffix, clip_bbox in crop_coords:
        clip_patch = dict()
        clip_patch['bbox'] = clip_bbox
        clip_patch['suffix'] = clip_suffix
        
        clip_patch['heatmap'], clip_patch['priority_score'] = get_judge_value_without_noun_list(image, question, clip_bbox)
        queue.put(Prioritize(-clip_patch['priority_score'], clip_patch))
  return related_bbox_2, max_patch['bbox']

ans_tot = 0
ans_ac = 0
ans_wa = 0
ans_uke = 0


# def encode_image(image):
#     # Create a BytesIO object to hold the image in memory
#     buffered = io.BytesIO()
#     # Save the image as JPEG (or PNG) to the BytesIO object
#     image.save(buffered, format="JPEG")  # Change format as needed
#     # Get the byte data and encode it to base64
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")


def crop_image(image: Image.Image, max_pixels: int = 1e5) -> Image.Image:
    width, height = image.size
    total_pixels = width * height

    if total_pixels <= max_pixels:
        return image

    ratio = (max_pixels / total_pixels) ** 0.5
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)

def save_cropped_image(image: Image.Image, filename: str):
    file_path = filename
    image.save(file_path)
    return file_path

record_of_test = []

def judge_qwen_api(item, image_dir, image2_bbox, image3_bbox):
  global record_of_test
  global filename 
  print("item: ", item)
  print("image_dir: ", image_dir)
  print("image2_bbox: ", image2_bbox)
  print("image3_bbox: ", image3_bbox)

  id = item["Question_id"]
  # image = item["Image"]
  question = item["Text"]
  choises = item["Answer choices"]
  ground_truth = item["Ground truth"]
  category = item["Category"]

  image = Image.open(image_dir).convert('RGB')
  
  related_bbox, pos_cue = position_cue_qwen(question, image.width, image.height)


  image3_bbox = [(image3_bbox[0]*2 + image2_bbox[0])//3, (image3_bbox[1]*2 + image2_bbox[1])//3, (image3_bbox[2]*2 + image2_bbox[2])//3, (image3_bbox[3]*2 + image2_bbox[3])//3] # Extend the bounding box by 1/3 in each direction

  # image2_bbox_01 = [image2_bbox[0] * 1.00 / image.width, image2_bbox[1] * 1.00 / image.height, image2_bbox[2] * 1.00 / image.width, image2_bbox[3] * 1.00 / image.height]
  # image3_bbox_01 = [image3_bbox[0] * 1.00 / image.width, image3_bbox[1] * 1.00 / image.height, image3_bbox[2] * 1.00 / image.width, image3_bbox[3] * 1.00 / image.height]

  # image2_ori = Image.open(image_dir).crop(image2_bbox).convert('RGB')
  # image3_ori = Image.open(image_dir).crop(image3_bbox).convert('RGB')
  # reasked_question = remove_position_cue_qwen(question)
  
  print("image2: ", image2_bbox)
  print("image3: ", image3_bbox)
  

  # resized_image_1 = crop_image(image)
  # file_path_1 = save_cropped_image(resized_image_1, 'resized_image_1.jpg')

  # resized_image_2 = crop_image(image2_ori)
  # file_path_2 = save_cropped_image(resized_image_2, 'resized_image_2.jpg')

  # resized_image_3 = crop_image(image3_ori)
  # file_path_3 = save_cropped_image(resized_image_3, 'resized_image_3.jpg')

  record_of_test.append([id, image_dir, ground_truth, category, image2_bbox, image3_bbox])

  with open(filename, 'w') as f:
    json.dump(record_of_test, f, indent=4) 
  torch.cuda.empty_cache()

def main():
  global model
  global filename 
  global client
  parser = argparse.ArgumentParser()
  parser.add_argument("--WEIGHTS_PATH", type=str, required=True)
  parser.add_argument("--json_dir", type=str, required=True)
  parser.add_argument("--CONFIG_PATH", type=str, required=True)
  parser.add_argument("--HOME", type=str, required=True)
  parser.add_argument("--API_KEY", type=str, required=True)

  args = parser.parse_args()

  print(args)

  WEIGHTS_PATH = args.WEIGHTS_PATH
  json_dir = args.json_dir
  CONFIG_PATH = args.CONFIG_PATH
  HOME = args.HOME
  API_KEY = args.API_KEY

  print("WEIGHTS_PATH: ", WEIGHTS_PATH)
  print("json_dir: ", json_dir)
  print("CONFIG_PATH: ", CONFIG_PATH)
  print("HOME: ", HOME)
  print("API_KEY: ", API_KEY)
  
  model = load_model(CONFIG_PATH, WEIGHTS_PATH)
  device = torch.device("cpu")
  model = model.to(device)



  with open(json_dir, 'r') as file:
      questions_box = json.load(file)

  with open(json_dir, 'r') as file:
      json_data = json.load(file)


  os.environ["QWEN_API_KEY"] = os.environ["DASHSCOPE_API_KEY"] = API_KEY

  client = OpenAI(
      api_key=os.getenv("DASHSCOPE_API_KEY"),
      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  )


  completion = client.chat.completions.create(
      model="qwen-plus",
      messages=[
          {'role': 'system', 'content': 'You are a helpful assistant.'},
          {'role': 'user', 'content': 'Who are you?'}],
      )
  response_content = completion.choices[0].message.content
  print(response_content)


  temp_questions_box = []
  list_of_category = []

  for json_data in questions_box:
    image_path = os.path.join(HOME, json_data.get("Image"))
    # print(image_path)
    if os.path.exists(image_path):
      try:
          with Image.open(image_path) as img:
              # img.crop([0,0,1,1])
              temp_questions_box.append(json_data)
              print(f"Processing {image_path}")
      except (IOError, UnidentifiedImageError):
          print(f"Can't Processing {image_path}")
      except OSError as e:
          print(f"File Damaged {image_path} - {e}")
  random.shuffle(temp_questions_box)


  existed_questions_box = []

  # print(temp_questions_box)

  list_of_category.append("color")
  list_of_category.append("count")
  list_of_category.append("position")

  # for json_data in temp_questions_box:
  #   if json_data.get("Category") not in list_of_category:
  #     list_of_category.append(json_data.get("Category"))


  for category in list_of_category:
    # print(category)
    cnt_existed = 0
    for json_data in temp_questions_box:
      if json_data.get("Category") == category and cnt_existed +1 <= 5:
        image_path = os.path.join(HOME, json_data.get("Image"))
        cnt_existed += 1
        if os.path.exists(image_path):
          existed_questions_box.append(json_data)



  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = f'record_of_test.json'

  for json_data in tqdm(existed_questions_box, desc="Processing questions", unit="question"):
    image_path = os.path.join(HOME, json_data.get("Image"))
    # print(image_path)
    if os.path.exists(image_path):
      print(image_path)

      image = Image.open(image_path)
      related_bbox_2, related_bbox_3 = Simple_VG(image_path, json_data.get('Text'))
      judge_qwen_api(json_data, image_path, related_bbox_2, related_bbox_3)
    else:
      print("File does not exist.")

if __name__ == "__main__":
    main()
