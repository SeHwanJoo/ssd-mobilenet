import os
import shutil
import numpy as np
import tensorflow as tf
import cv2

from core.make_dataset import TFDataset
from configuration import OBJECT_CLASSES, save_model_dir, test_picture_dir, NUM_CLASSES
from core.inference import InferenceProcedure
from core.ssd import SSD, ssd_prediction
from utils.tools import preprocess_image
from core.ground_truth import ReadDataset


def print_model_summary(network, shape):
    network.build_graph(shape)
    network.summary()


def find_class_name(class_id):
    for k, v in OBJECT_CLASSES.items():
        if v == class_id:
            return k
        elif class_id == 0:
            return 'no'


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            print('start with GPU 4')
            tf.config.experimental.set_visible_devices(gpus[4], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[4], True)
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)

    predicted_dir_path = '../mAP/predicted'
    ground_truth_dir_path = '../mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists("./dataset/detection/"): shutil.rmtree("./dataset/detection/")

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir("./dataset/detection/")

    ssd_model = SSD()
    print_model_summary(network=ssd_model, shape=(None, 224, 224, 3))
    ssd_model.load_weights(filepath=save_model_dir + "vgg16-epoch-49.h5")
    dataset = TFDataset()
    test_data, train_count = dataset.generate_datatset()


    # image = test_single_picture(picture_dir=test_picture_dir, model=ssd_model)

    def test_step(batch_images, batch_labels, name):
        print('=> ground truth of %s:' % name[0])
        ground_truth_path = os.path.join(ground_truth_dir_path, str(name[0]) + '.txt')
        with open(ground_truth_path, 'w') as f:
            i = 0
            while batch_labels[0][i][4] != -1:
                class_name = str(find_class_name(batch_labels[0][i][4]))
                xmin, ymin, xmax, ymax = list(map(str, batch_labels[0][i][:4]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
                i += 1

        print('=> predict result of %s:' % name[0])
        image_tensor = tf.expand_dims(batch_images[0], axis=0)
        procedure = InferenceProcedure(model=ssd_model)
        predict_result_path = os.path.join(predicted_dir_path, str(name[0]) + '.txt')
        is_object_exist, boxes, scores, classes = procedure.get_final_boxes(image=image_tensor)
        with open(predict_result_path, 'w') as f:
            if is_object_exist:
                for i in range(0, boxes.shape[0]):
                    coor = list(v.numpy() for v in boxes[i])
                    score = scores[i].numpy()
                    class_name = find_class_name(classes[i])
                    score = '%.4f' % score
                    # print(score)
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
            else:
                bbox_mess = ''
            f.write(bbox_mess)
            print('\t' + str(bbox_mess).strip())


    for step, batch_data in enumerate(test_data):
        images, labels, name = ReadDataset().read(batch_data)
        test_step(batch_images=images, batch_labels=labels, name=name)

    # with open("./dataset/voc2007_test.txt", 'r') as annotation_file:
    #     for num, line in enumerate(annotation_file):
    #         annotation = line.strip().split()
    #         image_path = annotation[0]
    #         image_name = image_path.split('/')[-1]
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         bbox_data_gt = get_bbox_data(image_name[:-4])
    #
    #         if len(bbox_data_gt) == 0:
    #             bboxes_gt = []
    #             classes_gt = []
    #         else:
    #             bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
    #         ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')
    #
    #         print('=> ground truth of %s:' % image_name)
    #         num_bbox_gt = len(bboxes_gt)
    #         with open(ground_truth_path, 'w') as f:
    #             for i in range(num_bbox_gt):
    #                 class_name = str(find_class_name(classes_gt[i]))
    #                 xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
    #                 bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
    #                 f.write(bbox_mess)
    #                 print('\t' + str(bbox_mess).strip())
    #         print('=> predict result of %s:' % image_name)
    #         predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
    #         # Predict Process
    #         image_tensor = preprocess_image(image_path)
    #         image_tensor = tf.expand_dims(image_tensor, axis=0)
    #         procedure = InferenceProcedure(model=ssd_model)
    #         is_object_exist, boxes, scores, classes = procedure.get_final_boxes(image=image_tensor)
    #         if is_object_exist:
    #             with open(predict_result_path, 'w') as f:
    #                 for i in range(0, len(classes)):
    #                     coor = list(v.numpy() for v in boxes[i])
    #                     score = scores[i].numpy()
    #                     class_name = find_class_name(classes[i])
    #                     # print(classes[i])
    #                     score = '%.4f' % score
    #                     # print(coor, score, class_name)
    #                     xmin, ymin, xmax, ymax = list(map(str, coor))
    #                     bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
    #                     f.write(bbox_mess)
    #                     print('\t' + str(bbox_mess).strip())
