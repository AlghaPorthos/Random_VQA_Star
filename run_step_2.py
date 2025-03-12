import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dashscope
import random
from tqdm import tqdm
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
import argparse



def crop_image(image: Image.Image, max_pixels: int = 5e5) -> Image.Image:
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


categories = []
for entry in data:
    category = entry[3]
    if category not in categories:
        categories.append(category)

print(categories)

# categories = ['position', 'vehicle/counting', 'Attribute_Motion_MultiPedestrians', 'Relation_Interaction_Other2Other', 'vehicle/attribute/orientation', 'color', 'Prediction_Intention_Pedestrian', 'Relation_Interaction_Ego2Pedestrain', 'count', 'vehicle/location', 'Attribute_Motion_MultiVehicles', 'Objects_Identify', 'Person/counting', 'person/counting', 'Object_Count', 'vehicle/attribute/color', 'Attribute_Motion_Pedestrain', 'Attention_TrafficSignal', 'Prediction_Intention_Ego', 'Attribute_Visual_TrafficSignal', 'person/attribute/color', 'calculate', 'property', 'Vehicle/counting', 'person/attribute/orientation', 'Relation_Interaction_Ego2Vehicle', 'Prediction_Intention_Vehicle', 'Relation_Interaction_Ego2TrafficSignal', 'Attribute_Motion_Vehicle', 'intention']
ans_ac_1 = {category: 0 for category in categories}
ans_wa_1 = {category: 0 for category in categories}
ans_ac_2 = {category: 0 for category in categories}
ans_wa_2 = {category: 0 for category in categories}
ans_ac_3 = {category: 0 for category in categories}
ans_wa_3 = {category: 0 for category in categories}
ans_ac_4 = {category: 0 for category in categories}
ans_wa_4 = {category: 0 for category in categories}


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_qwen_response(item, file_path_1, file_path_2, file_path_3, image2_bbox, image3_bbox, question = None, reasked_question = None):
#   return "Z", "WA" 
  id = item["Question_id"]
  # image = item["Image"]
  if question is None:
    question = item["Text"]
  choises = item["Answer choices"]
  ground_truth = item["Ground truth"]
  category = item["Category"]
  if file_path_1 is not None:
    if file_path_2 is None and file_path_3 is None:
      messages_1 = [
          {
              "role": "user",
              "content": [
                  {"image": file_path_1},
                  # {"image": file_path_2},
                  # {"image": file_path_3},
                  {"text": f"{question} The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
              ]
          }
      ]
    elif file_path_2 is None and file_path_3 is not None:
      messages_1 = [
          {
              "role": "user",
              "content": [
                  {"image": file_path_1},
                  # {"image": file_path_2},
                  {"image": file_path_3},
                  {"text": f"{question} The two images are shown, the first image is the original one, and the other is a more detailed look. The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
                #   {"text": f"{question} The two images are shown, the first image is the original one, and the other is a more detailed look, which coordinats is {image3_bbox}. The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
              ]
          }
      ]
    elif file_path_2 is not None and file_path_3 is None:
      messages_1 = [
          {
              "role": "user",
              "content": [
                  {"image": file_path_1},
                  {"image": file_path_2},
                  # {"image": file_path_3},
                  {"text": f"{question} The two images are shown, the first image is the original one, and the other is a more detailed look. The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
                #   {"text": f"{question} The two images are shown, the first image is the original one, and the other is a more detailed look, which coordinate is {image2_bbox}. The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
              ]
          }
      ]
    elif file_path_2 is not None and file_path_3 is not None:
      messages_1 = [
          {
              "role": "user",
              "content": [
                  {"image": file_path_1},
                  {"image": file_path_2},
                  {"image": file_path_3},
                  {"text": f"{question} The three images are shown, the first image is the original one, and the other is a more detailed look, which coordinates are {image2_bbox} and {image3_bbox}, respectively. The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
              ]
          }
      ]
  else:
    
    if file_path_2 is None and file_path_3 is None:
      return "Z", "WA"
      # messages_1 = [
      #     {
      #         "role": "user",
      #         "content": [
      #             # {"image": file_path_1},
      #             # {"image": file_path_2},
      #             # {"image": file_path_3},
      #             {"text": f"{question} The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
      #         ]
      #     }
      # ]
    elif file_path_2 is None and file_path_3 is not None:
      messages_1 = [
          {
              "role": "user",
              "content": [
                  # {"image": file_path_1},
                  # {"image": file_path_2},
                  {"image": file_path_3},
                  {"text": f"{reasked_question} The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
              ]
          }
      ]
    elif file_path_2 is not None and file_path_3 is None:
      messages_1 = [
          {
              "role": "user",
              "content": [
                  # {"image": file_path_1},
                  {"image": file_path_2},
                  # {"image": file_path_3},
                  {"text": f"{reasked_question} The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
              ]
          }
      ]
    elif file_path_2 is not None and file_path_3 is not None:
      messages_1 = [
          {
              "role": "user",
              "content": [
                  # {"image": file_path_1},
                  {"image": file_path_2},
                  {"image": file_path_3},
                  {"text": f"{reasked_question} The two images are shown, the first image is the original one, and the other is a more detailed look. The choices are listed below: {choises}\n Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. The best answer is:"}
              ]
          }
      ]
  

  response_1 = dashscope.MultiModalConversation.call(
      api_key=os.getenv('DASHSCOPE_API_KEY'),
      model='qwen-vl-max-latest',
      messages=messages_1,
      vl_high_resolution_images=False
  )
  try:
    if response_1 is None:
      print("API Failed")
      return "Z", "WA"
  except Exception as e:
    print(f"Error: {e}")
    return "Z", "WA"
  try:
    response_1 = response_1.output.choices[0].message.content[0]["text"]
  except Exception as e:
    print(f"Error: {e}")
    return "Z", "WA"
  print("Response: ",response_1)

  # print(sentence)

  print(response_1)
  if ground_truth == response_1:
    judge_1 = "AC_1"
  # elif ground_truth in response_1:
  #   judge_1 = "AC_2a"
  elif response_1 in ground_truth:
    judge_1 = "AC_2b"
  else:
    cor = 0
    for ANS in ["A", "B", "C", "D"]:
      if ANS in ground_truth and ANS in response_1:
        cor += 1
    if cor == 1:
      judge_1 = "AC_3"
    elif cor > 1:
      judge_1 = "PE_2"
    else:
      judge_1 = "WA"
  return response_1, judge_1

async def async_get_qwen_response(executor, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, get_qwen_response, *args)

def position_cue_qwen(sentence, w = 1.00, h = 1.00):
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


        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You will be given a question, Please give me the most related region according to the position cue the question asked. Please select from the following word WITHOUT any other word: right edge, left edge, top edge, bottom edge, right, left, top, bottom, bottom right corner, bottom left corner, top left corner, top right corner, lower right corner, lower left corner, upper left corner, upper right corner, bottom right, bottom left, top left, top right, lower right, lower left, upper left, upper right, central, middle, middle right, middle left, upper middle, lower middle, top middle, bottom middle, full frame. If there is none, use \'full frame\''},
                {'role': 'user', 'content': sentence}],
            )
        response_content = completion.choices[0].message.content
        print(response_content)

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
  for i in range(1):
    related_bbox, pos_cue = position_cue_qwen(sentence)
    if pos_cue == 'full frame':
      # print(sentence)
      return sentence
    else:
      print("Re-generating: " + str(i+1) + " of 10")
    # user_msg = "Here is a question: " + sentence + "This question has attention of " + pos_cue + " of the image. But acturally the image may be cropped in the process, so any position cue based on the whole image is invalid. Now please remove the position cue related to " + pos_cue + " based on the whole image (note that this is not paraphrasing), but keep OTHER information all the same."

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': "You will be given a question, This question has attention of " + pos_cue + " of the image. But acturally the image may be cropped in the process, so any position cue based on the whole image is invalid. Now please remove the position cue related to " + pos_cue + " based on the whole image (note that this is not paraphrasing), but keep OTHER information all the same."},
            {'role': 'user', 'content': sentence}],
        )
    sentence = completion.choices[0].message.content
    print(sentence)
  return sentence
async def main():
  executor = ThreadPoolExecutor()
  random.shuffle(data)

  parser = argparse.ArgumentParser()
  parser.add_argument("--json_file_path", type=str, required=True)
  parser.add_argument("--original_json_file_path", type=str, required=True)
  parser.add_argument("--image_directory", type=str, required=True)
  parser.add_argument("--list_of_history_dir", type=str, required=True)
  parser.add_argument("--API_KEY", type=str, required=True)


  args = parser.parse_args()
  print(args)

  json_file_path = args.json_file_path
  original_json_file_path = args.original_json_file_path
  image_directory = args.image_directory
  list_of_history_dir = args.list_of_history_dir
  API_KEY = args.API_KEY

  print("json_file_path: ", json_file_path)
  print("original_json_file_path: ", original_json_file_path)
  print("image_directory: ", image_directory)
  print("list_of_history_dir: ", list_of_history_dir)
  print("API_KEY: ", API_KEY)


  

  # json_file_path = os.environ.get("json_file_path")
  # original_json_file_path = os.environ.get("original_json_file_path")
  # image_directory = os.environ.get("image_directory")
  # API_KEY = os.environ.get("API_KEY")



  # os.environ["DASHSCOPE_API_KEY"] = API_KEY


  with open(json_file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)

  with open(original_json_file_path, 'r', encoding='utf-8') as f:
      original_data = json.load(f)

  with open(list_of_history_dir, 'r', encoding='utf-8') as f:
      list_of_history = json.load(f)

  for entry in tqdm(data, desc="Processing entries", unit="entry"):    
      image_id, question, reasked_question, category_in_json, response_1, judge_1, response_2, judge_2, response_3, judge_3, response_4, judge_4 = None, None, None, None, None, None, None, None, None, None, None, None
      image_id = entry[0]
      flag = False
      for history in list_of_history:
        history_id = history[0]
        if image_id == history_id:
          print(history)
          flag = True
          image_id, question, reasked_question, category_in_json, response_1, judge_1, response_2, judge_2, response_3, judge_3, response_4, judge_4 = history
          break
      if flag == False:
        image_file = None
        question = None
        choises = None
        ground_truth = None
        matched_json_data = None
        for json_data in original_data:
            id = json_data.get("Question_id")
            # if id.rsplit('/', 1)[-1] == image_id.rsplit('/', 1)[-1]:
            if id == image_id:
                print("found id: ", id)
                matched_json_data = json_data
                image = json_data.get("Image")
                choises = json_data.get("Answer choices")
                category_in_json = json_data.get("Category")
                image_file = os.path.join(image_directory, image)
                question = json_data.get("Text")
                ground_truth = json_data.get("Ground truth")
                break
            else:
              pass
                # print(id.rsplit('/', 1)[-1], image_id.rsplit('/', 1)[-1])
        if category_in_json in categories:
          pass
        else:
          print(category_in_json, "is not in categories")
          continue
        if os.path.exists(image_file):
            print("Detected: ", image_file)
            image = Image.open(image_file)
            # draw = ImageDraw.Draw(image)

            image2_bbox = entry[4]  # image2_bbox
            image3_bbox = entry[5]  # image3_bbox

            
            image = Image.open(image_file)


            # rect1 = patches.Rectangle((image2_bbox[0], image2_bbox[1]), 
            #                         image2_bbox[2] - image2_bbox[0], 
            #                         image2_bbox[3] - image2_bbox[1], 
            #                         linewidth=2, edgecolor='red', facecolor='none')
            # rect2 = patches.Rectangle((image3_bbox[0], image3_bbox[1]), 
            #                         image3_bbox[2] - image3_bbox[0], 
            #                         image3_bbox[3] - image3_bbox[1], 
            #                         linewidth=2, edgecolor='blue', facecolor='none')
            # fig, ax = plt.subplots(1)
            # ax.imshow(image)
            # ax.add_patch(rect1)
            # ax.add_patch(rect2)
            # plt.title(question + "\n" + choises[0] + "\n" + choises[1] + "\n" + choises[2] + "\n" + choises[3] + "\n" + choises[4] + ground_truth)
            # plt.show()
            

            image2_ori = Image.open(image_file).crop(image2_bbox).convert('RGB')
            image3_ori = Image.open(image_file).crop(image3_bbox).convert('RGB')
            reasked_question = remove_position_cue_qwen(question)
            
            print("image2: ", image2_bbox)
            print("image3: ", image3_bbox)
            
            image2_bbox_01 = [image2_bbox[0] * 1.00 / image.width, image2_bbox[1] * 1.00 / image.height, image2_bbox[2] * 1.00 / image.width, image2_bbox[3] * 1.00 / image.height]
            image3_bbox_01 = [image3_bbox[0] * 1.00 / image.width, image3_bbox[1] * 1.00 / image.height, image3_bbox[2] * 1.00 / image.width, image3_bbox[3] * 1.00 / image.height]


            resized_image_1 = crop_image(image)
            file_path_1 = save_cropped_image(resized_image_1, 'resized_image_1.jpg')

            resized_image_2 = crop_image(image2_ori)
            file_path_2 = save_cropped_image(resized_image_2, 'resized_image_2.jpg')

            resized_image_3 = crop_image(image3_ori)
            file_path_3 = save_cropped_image(resized_image_3, 'resized_image_3.jpg')


            tasks = [
                async_get_qwen_response(executor, matched_json_data, file_path_1, None, None, None, None, question, None),
                async_get_qwen_response(executor, matched_json_data, file_path_1, None, file_path_3, None, image3_bbox_01, question, reasked_question),
                async_get_qwen_response(executor, matched_json_data, file_path_1, file_path_2, None, image2_bbox_01, None, question, reasked_question),
                async_get_qwen_response(executor, matched_json_data, file_path_1, file_path_2, file_path_3, image2_bbox_01, image3_bbox_01, question, reasked_question),
            ]

            # 等待所有任务完成
            results = await asyncio.gather(*tasks)

            # 解析结果
            response_1, judge_1 = results[0]
            response_2, judge_2 = results[1]
            response_3, judge_3 = results[2]
            response_4, judge_4 = results[3]
            # response_1, judge_1 = get_qwen_response(matched_json_data, file_path_1, None, None, None, None, question, None)
            # response_2, judge_2 = get_qwen_response(matched_json_data, file_path_1, None, file_path_3, None, image3_bbox_01, question, reasked_question)
            # response_3, judge_3 = get_qwen_response(matched_json_data, file_path_1, file_path_2, None, image2_bbox_01, None, question, reasked_question)
            # response_4, judge_4 = get_qwen_response(matched_json_data, file_path_1, file_path_2, file_path_3, image2_bbox_01, image3_bbox_01, question, reasked_question)

      # response_1, judge_1 = "Z", "WA"
      if "AC" in judge_1:
          ans_ac_1[category_in_json] += 1
      else:
          ans_wa_1[category_in_json] += 1
      # response_2, judge_2 = "Z", "WA"
      if "AC" in judge_2:
          ans_ac_2[category_in_json] += 1
      else:
          ans_wa_2[category_in_json] += 1
      # response_3, judge_3 = "Z", "WA"
      if "AC" in judge_3:
          ans_ac_3[category_in_json] += 1
      else:
          ans_wa_3[category_in_json] += 1
      # response_3, judge_3 = "Z", "WA"
      if "AC" in judge_4:
          ans_ac_4[category_in_json] += 1
      else:
          ans_wa_4[category_in_json] += 1
      print("Situation_1: ", response_1, judge_1)
      print("Situation_2: ", response_2, judge_2)
      print("Situation_3: ", response_3, judge_3)
      print("Situation_4: ", response_4, judge_4)
      # if "WA" in judge_1 and "AC" in judge_2 and "WA" in judge_3:
      #   print("GET:                                                                              ", image_id)
      for category in categories:
          print("-------- Category: ", category, "--------")
          print("Situation_1, AC: ", ans_ac_1[category], "WA: ", ans_wa_1[category], "TOT: ", ans_ac_1[category] + ans_wa_1[category])
          print("Situation_2, AC: ", ans_ac_2[category], "WA: ", ans_wa_2[category], "TOT: ", ans_ac_2[category] + ans_wa_2[category])
          print("Situation_3, AC: ", ans_ac_3[category], "WA: ", ans_wa_3[category], "TOT: ", ans_ac_3[category] + ans_wa_3[category])
          print("Situation_4, AC: ", ans_ac_4[category], "WA: ", ans_wa_4[category], "TOT: ", ans_ac_4[category] + ans_wa_4[category])
      # print(entry[2], ac1, ac2, ac3)
          
      # else:
      #     print(f"Image file not found: {image_file}")"
      list_of_history.append([image_id, question, reasked_question, category_in_json, response_1, judge_1, response_2, judge_2, response_3, judge_3, response_4, judge_4])
      # dump list_of_history to .json file
      with open(list_of_history_dir, 'w') as f:
        json.dump(list_of_history, f)

asyncio.run(main())
