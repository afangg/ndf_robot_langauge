import torch
import numpy as np
import open3d as o3d

from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from scipy import stats
import matplotlib.pyplot as plt

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from rndf_robot.utils import util, trimesh_util
from airobot import Robot, log_info, set_log_level, log_warn, log_debug


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

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

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
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


def owlvit_detect(image, classes, show_seg=False):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    obj_ids = list(classes.keys())
    # how does it handle ['a photo of a mug', 'a photo of a bowl', 'a photo of a mug']
    texts = [[f'a photo of {classes[obj_id]}' for obj_id in obj_ids]]

    pil_image = Image.fromarray(np.uint8(image)).convert('RGB')
    inputs = processor(text=texts, images=pil_image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([pil_image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    texts = texts[i]
    bounding_boxes = {}
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    score_threshold = 0.05

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            confidence_score = round(score.item(), 3)
            if obj_ids[label] not in bounding_boxes or confidence_score > bounding_boxes[obj_ids[label]][0]:
                bounding_boxes[obj_ids[label]] = (confidence_score, box)

    if show_seg:
        for _, box_score in bounding_boxes.items():
            score, box = box_score
            xmin,ymin,xmax,ymax = box
            print(f"Detected {texts[label]} with confidence {confidence_score} at location {box}")
            draw_bounding_box_on_image(pil_image, ymin, xmin, ymax, xmax, color=np.random.choice(STANDARD_COLORS), 
                                       display_str_list=[f"{texts[label]}: {score}"], use_normalized_coordinates=False)
        plt.imshow(pil_image)
        plt.show()
    return bounding_boxes

def detect_bbs(image, classes):
    ids_to_bb = owlvit_detect(image, classes)
    obj_to_region = {}
    for obj_id, score_box in ids_to_bb.items():
        box = score_box[1]
        obj_to_region[obj_id] = (list(int(i) for i in box))
        # cropped_img = image[ymin:ymax,xmin:xmax,:]
    return obj_to_region

def get_largest_pcd(pcd, show_scene=False):
    region_pcd = o3d.geometry.PointCloud()
    region_pcd.points = o3d.utility.Vector3dVector(pcd)
    labels = np.array(region_pcd.cluster_dbscan(eps=0.015, min_points=20))
    freq_label = stats.mode(labels)[0]
    max_pcd = pcd[np.where(labels == freq_label)]
    if show_scene:
        pcds = []
        for label in set(labels):
            label_pcd = pcd[np.where(labels == label)]
            pcds.append(label_pcd)
        trimesh_util.trimesh_show(pcds)
    return max_pcd      

def get_region(full_pcd, region):
    xmin,ymin,xmax,ymax, = region
    cropped_pcd = full_pcd[ymin:ymax,xmin:xmax,:]
    return cropped_pcd.reshape((-1, 3))

def get_obj_pcds(rgb, pts_raw, obj_classes):
    obj_pcds = {}
    height, width, _ = rgb.shape
    pts_2d = pts_raw.reshape((height, width, 3))

    obj_regions = detect_bbs(rgb, obj_classes)

    for obj_class in obj_regions:
        for i, region in enumerate(obj_regions[obj_class]):
            region_pcd = get_region(pts_2d, region)
            largest_cluster = get_largest_pcd(region_pcd)
            z = largest_cluster[:, 2]
            min_z = z.min()

            table_mask = np.where(z <= min_z+0.001)
            obj_mask = np.where(z > min_z)
            obj_pcd = largest_cluster[obj_mask]

            # table_z_max, table_z_min =  
            if obj_class not in obj_pcds:
                obj_pcds[obj_class] = []
            obj_pcds[obj_class].append(obj_pcd)
    return obj_pcds