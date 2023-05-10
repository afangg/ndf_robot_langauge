import torch
import numpy as np
import open3d as o3d

from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

import matplotlib.pyplot as plt

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from airobot import log_debug

class OWLViTDetect:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model.eval()

    def detect_captions(self, image, descriptions, top=None, score_threshold=0.05):
        '''
        Detects objects in image that corresponding to a given description and returns the bounding boxes
        of the parts of the image that match above a certain threshold

        image: numpy (n,m,3) rgb image
        descriptions: list of obj descriptions to detect

        return: {description: [bb1, bb2]}
        '''
        
        texts = [[f'a photo of a {description}' for description in descriptions]]

        pil_image = Image.fromarray(np.uint8(image))
        inputs = self.processor(text=texts, images=pil_image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([pil_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        texts = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        bbox_and_score = {}
        log_debug(f"Highest score is {max(scores)}")

        for box, score, caption in zip(boxes, scores, labels):
            if score < score_threshold:
                continue
            
            label = descriptions[caption]
            if label not in bbox_and_score:
                bbox_and_score[label] = []

            box = [int(i) for i in box.tolist()]
            score = round(score.item(),3)
            bbox_and_score[label].append((box, [score]))

        out_boxes = {}
        out_scores = {}
        for label, box_and_scores in bbox_and_score.items():
            sorted_box_scores = sorted(box_and_scores, key=lambda x: x[1], reverse=True)
            max_count = top if top is not None else len(sorted_box_scores)
        
            top_box_scores = sorted_box_scores[:max_count]
            out_boxes[label] = [box_score[0] for box_score in top_box_scores] 
            out_scores[label] = [box_score[1] for box_score in top_box_scores] 

        return out_boxes, out_scores
    
    def show_owl(self, image, captions_to_bboxes, bbox_scores):
      pil_image = Image.fromarray(np.uint8(image))
      for label, bboxes in captions_to_bboxes.items():
          for i in range(len(bboxes)):
              box, score = bboxes[i], bbox_scores[label][i]
              xmin,ymin,xmax,ymax = box
              score = score[0]
              draw_bounding_box_on_image(pil_image, ymin, xmin, ymax, xmax, display_str_list=[f"{label}: {score}"], use_normalized_coordinates=False)
      plt.imshow(pil_image)
      plt.show()

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
      (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    else:
      (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
              (right, top), (left, top)], width=thickness, fill=color)
    try:
      font = ImageFont.truetype(font="/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=15)
    except IOError:
      font = ImageFont.load_default()

    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
      text_bottom = top
    else:
      text_bottom = bottom + total_display_str_height
    for display_str in display_str_list[::-1]:
      text_left = min(5, left)
      text_width, text_height = font.getsize(display_str)
      margin = np.ceil(0.05 * text_height)
      draw.rectangle(
          [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                            text_bottom)],
          fill=color)
      draw.text(
          (left + margin, text_bottom - text_height - margin),
          display_str,
          fill='black',
          font=font)
      text_bottom -= text_height - 2 * margin
