import torch
import numpy as np
import open3d as o3d

from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

import matplotlib.pyplot as plt
from rndf_robot.utils import util, trimesh_util

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from airobot import log_debug
from IPython import embed;

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


def owlvit_detect(image, descriptions, score_threshold=0.05, show_seg=False):
    '''
    Detects objects in image that corresponding to a given description and returns the bounding boxes
    of the parts of the image that match above a certain threshold

    image: numpy (n,m,3) rgb image
    descriptions: list of obj descriptions to detect

    return: {description: [bb1, bb2]}
    '''
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    # how does it handle ['a photo of a mug', 'a photo of a bowl', 'a photo of a mug']
    texts = [[f'a photo of a {description}' for description in descriptions]]

    pil_image = Image.fromarray(np.uint8(image))
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
    highest_scores = {}

    log_debug(f"Highest score is {max(scores)}")

    for box, score, caption in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        label = descriptions[caption]

        if score < score_threshold:
            continue
        # print(f"Detected {label} with confidence {score} at location {box}")

        if label not in bounding_boxes:
            bounding_boxes[label] = []
            highest_scores[label] = []
        
        # what to do if two objs of the same class are on top of each other lol (one in front of the other)
        # just remove larger one? idk
        remove_idx = None
        for i, existing_bb in enumerate(bounding_boxes[label]):
            smaller = bb_contained(box, existing_bb) 
            if smaller == 0: # this new bb is bigger, don't add it
                remove_idx = -1
            elif smaller == 1: # this new bb is smaller and overlaps, remove the bigger one
                remove_idx = i
            else: # they dont overlap, continue
                continue
        score = round(score.item(),3)
        bounding_boxes[label].append(box)
        highest_scores[label].append([score])
        if remove_idx is not None:
            del bounding_boxes[label][remove_idx]
            del highest_scores[label][remove_idx]

        # obj_captions[label]
        # for existing_bb in obj_captions[label]:
        #     not_contained = False
        #     for i in range(4):
        #         if box[i] >= existing_bb[i]:
        #             not_contained = True
        #             break
        #     if not_contained: continue

    if show_seg:
        for label in bounding_boxes:
            bbs = bounding_boxes[label]
            scores = highest_scores[label]
            for i in range(len(bbs)):
                xmin,ymin,xmax,ymax = bbs[i]
                # score = round(scores[i].item(), 3)
                score = scores[i]
                # print(f"Detected {label} with confidence {score} at location {bb}")
                # draw_bounding_box_on_image(pil_image, ymin, xmin, ymax, xmax, color=np.random.choice(STANDARD_COLORS), 
                #                         display_str_list=[f"{label}: {score}"], use_normalized_coordinates=False)
                draw_bounding_box_on_image(pil_image, ymin, xmin, ymax, xmax, color=np.random.choice(STANDARD_COLORS), 
                                display_str_list=[f"{label}: {score}"], use_normalized_coordinates=False)
        plt.imshow(pil_image)
        plt.show()

    label_count = {label: len(bounding_boxes[label]) for label in bounding_boxes}
    return bounding_boxes, highest_scores

def bb_contained(bb1, bb2, margin=0.01):
    '''
    bb1: (xmin,ymin,xmax,ymax)
    bb2: (xmin,ymin,xmax,ymax)
    return: 0, (bb1 in bb2), 1 (bb2 in bb1), None (bb not within eachother)
    '''
   
    for i in range(2):
        xmin1,ymin1,xmax1,ymax1 = bb1
        xmin2,ymin2,xmax2,ymax2 = bb2  

        if xmin1 >= xmin2-margin and xmax1 <= xmax2+margin and ymin1 >= ymin2-margin and ymax1 <= ymax2+margin:
            return i
        bb1, bb2 = bb2, bb1
    return None
   
def detect_bbs(image, classes):
    ids_to_bb, cam_scores = owlvit_detect(image, classes, score_threshold=0.2, show_seg=False)
    obj_to_region = {}
    for obj_id, boxes in ids_to_bb.items():
        obj_to_region[obj_id] = [(list(int(i) for i in box)) for box in boxes]
    return obj_to_region, cam_scores

def get_largest_pcd(pcd, show_scene=False):
    region_pcd = o3d.geometry.PointCloud()
    region_pcd.points = o3d.utility.Vector3dVector(pcd)
    labels = np.array(region_pcd.cluster_dbscan(eps=0.015, min_points=10))
    # if len(labels) == 0: return np.array([])
    freq_label = mode(labels)[0][0]
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
    flat_pcd = cropped_pcd.reshape((-1, 3))
    return flat_pcd

def get_obj_pcds(rgb, pts_raw, obj_classes):
    obj_pcds = {}
    height, width, _ = rgb.shape
    pts_2d = pts_raw.reshape((height, width, 3))

    obj_regions = detect_bbs(rgb, obj_classes)
    for obj_class in obj_regions:
        for i, region in enumerate(obj_regions[obj_class]):
            region_pcd = get_region(pts_2d, region)
            largest_cluster = get_largest_pcd(region_pcd, show_scene=False)
            if largest_cluster.any(): continue
            z = largest_cluster[:, 2]
            min_z = z.min()

            table_mask = np.where(z <= min_z+0.001)
            obj_mask = np.where(z > min_z)
            obj_pcd = largest_cluster[obj_mask]
            trimesh_util.trimesh_show([obj_pcd])

            # table_z_max, table_z_min =  
            if obj_class not in obj_pcds:
                obj_pcds[obj_class] = []
            obj_pcds[obj_class].append(obj_pcd)
    return obj_pcds

def extend_pcds(cam_pcds, pcd_list, cam_scores, pcd_scores, threshold=0.08):
    '''
    Given partial pcds in a camera view, finds which existing incomplete pcd it most likely belongs to
    Finds the partial pcd with the closest centroid that is at least 2cms away from the existing pcds

    Remaining partial pcds will be assumed to belong to a new obj and a new set is created

    cam_pcds: np.array (n, m, 3)
    pcd_list: np.array (j, k, 3)
    cam_scores: (n, a) list
    pcd_scores: (j, b) list

    return: updated pcd list
    '''

    #might want to keep each camera's pcd seperated
    centroids = [np.average(partial_obj, axis=0) for partial_obj in pcd_list]
    new_centroids = np.array([np.average(pcd, axis=0) for pcd in cam_pcds])

    # centroids = np.average(pcd_list, axis=1)
    # new_centroids = np.average(cam_pcds, axis=1)
    new_list = []
    for i, centroid in enumerate(centroids):
        if len(new_centroids) == 0: break
        diff = new_centroids-centroid
        centroid_dists = np.sqrt(np.sum(diff**2,axis=-1))
        min_idx = np.argmin(centroid_dists)
        log_debug(f'closest centroid is {centroid_dists[min_idx]} away')

        updated_pcds = pcd_list[i]
        if centroid_dists[min_idx] <= threshold:
            updated_pcds = np.concatenate((updated_pcds, cam_pcds[min_idx]))
            pcd_scores[i].append(cam_scores.pop(min_idx)[0])
            cam_pcds = np.delete(cam_pcds, min_idx, 0)
            new_centroids = np.delete(new_centroids, min_idx, 0)
        new_list.append(updated_pcds)
    return new_list + list(cam_pcds), pcd_scores + cam_scores

def get_label_pcds(regions, pts_2d):
    label_pcds = []
    for region in regions:
        region_pcd = get_region(pts_2d, region)
        largest_cluster = get_largest_pcd(region_pcd, True)
        z = largest_cluster[:, 2]
        min_z = z.min()

        table_mask = np.where(z <= min_z+0.01)
        obj_mask = np.where(z > min_z+0.01)
        obj_pcd = largest_cluster[obj_mask]
        label_pcds.append(obj_pcd)
    return label_pcds

def mode(a, axis=0):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts