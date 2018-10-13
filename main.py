import math
import time
import cv2
import numpy as np
import tensorflow as tf
from model import Model
import utils


def run():
    mod = Model()
    saver = tf.train.Saver()
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sess_config) as sess:
        print('Restoring from storage')
        saver.restore(sess, './data/cpm_hand')
        # stage_path = 'stage_1/stage_heatmap/BiasAdd:0'
        # stage_path = 'stage_2/mid_conv7/BiasAdd:0'
        stage_path = 'stage_3/mid_conv7/BiasAdd:0'
        # stage_path = 'sub_stages/sub_conv2/BiasAdd:0'
        output_node = tf.get_default_graph().get_tensor_by_name(stage_path)

        cam = cv2.VideoCapture(0)
        fps_tracker = utils.FPS_Tracker()
        while True:
            _, full_img = cam.read()
            # full_img = np.flip(full_img, axis=1)
            # cv2.imshow('frame', full_img)
            input_img = full_img.copy()
            input_img = utils.img_white_balance(input_img, 5)
            h = input_img.shape[0]//2 - 128
            w = input_img.shape[1]//2 - 128
            center_img = full_img[h:h+256, w:w+256]
            input_img = input_img / 256.0 - 0.5

            input_img = input_img[h:h+256, w:w+256]
            input_img = np.expand_dims(input_img, axis=0)

            stage_heatmap_np = sess.run(
                [output_node],
                feed_dict={mod.input_images: input_img}
            )
            
            sz = mod.input_size
            heatmap = stage_heatmap_np[0][0, :, :, 0:mod.joints]
            heatmap = cv2.resize(heatmap, (sz, sz))
            draw_hand(center_img, heatmap)
            heatmap = np.amax(heatmap, axis=2)
            heatmap = np.reshape(heatmap, (sz, sz, 1))
            heatmap = np.repeat(heatmap, 3, axis=2)
            heatmap *= 255
            # print(heatmap)

            cv2.imshow('frame', center_img)
            cv2.imshow('heatmap', heatmap.astype(np.uint8))
            fps_tracker.new_frame()

            if cv2.waitKey(1) & 0xFF == ord('q'): break

# Limb connections
limbs = [[0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [0, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [0, 17],
            [17, 18],
            [18, 19],
            [19, 20]
            ]

# Finger colors
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

def draw_hand(center_img, heatmap):
    num_of_joints = heatmap.shape[2]
    input_size = center_img.shape[0]
    input_shape = (input_size, input_size)
    joint_coords = np.zeros((num_of_joints, 2))

    # Plot joints
    for joint_num in range(num_of_joints):
        tmp_heatmap = heatmap[:, :, joint_num]
        joint_coord = np.unravel_index(np.argmax(tmp_heatmap), input_shape)
        joint_coord = np.array(joint_coord).astype(np.float32)
        joint_coords[joint_num, :] = joint_coord

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            cv2.circle(center_img, center=(int(joint_coord[1]), int(joint_coord[0])), radius=3,
                       color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            cv2.circle(center_img, center=(int(joint_coord[1]), int(joint_coord[0])), radius=3,
                       color=joint_color, thickness=-1)

    # Plot limbs
    for limb_num in range(len(limbs)):
        x1 = int(joint_coords[int(limbs[limb_num][0])][0])
        y1 = int(joint_coords[int(limbs[limb_num][0])][1])
        x2 = int(joint_coords[int(limbs[limb_num][1])][0])
        y2 = int(joint_coords[int(limbs[limb_num][1])][1])
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            cv2.fillConvexPoly(center_img, polygon, color=limb_color)


if __name__ == '__main__':
    run()

