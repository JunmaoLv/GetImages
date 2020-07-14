import tensorflow as tf
from configuration import save_model_dir, test_image_dir, model_index, EPOCHS, model_name_list, BATCH_SIZE
from prepare_data import load_and_preprocess_image
from train import get_model
import os
from timeit import default_timer as timer


def get_single_picture_prediction_from_camera(model, image):
    class_name_list = ['dirty_defect', 'edge_defect', 'linear_defect', 'no_defect', 'yarn_defect']
    image = tf.expand_dims(image, axis=0)
    prediction = model(image, training=False)
    pred_class_num = tf.math.argmax(prediction, axis=-1)
    pred_class_name = class_name_list[int(pred_class_num)]
    return pred_class_name


def get_single_picture_prediction(model, picture_dir):
    class_name_list = ['dirty_defect', 'edge_defect', 'linear_defect', 'no_defect', 'yarn_defect']
    picture_list = os.listdir(picture_dir)
    picture_path = os.path.join(picture_dir, picture_list[0])
    image_tensor = load_and_preprocess_image(tf.io.read_file(filename=picture_path), data_augmentation=False)
    image = tf.expand_dims(image_tensor, axis=0)
    prediction = model(image, training=False)
    pred_class_num = tf.math.argmax(prediction, axis=-1)
    pred_class_name = class_name_list[int(pred_class_num)]
    return pred_class_name
    # return prediction


def get_list_picture_prediction(model, picture_dir):
    class_name_list = sorted([s for s in os.listdir(picture_dir) if 'README' not in s])
    print('class name:')
    print(class_name_list)
    image_tensor_list = []
    for class_name_item in class_name_list:
        picture_path = os.path.join(picture_dir, class_name_item)
        picture_path_list = os.listdir(picture_path)
        for picture_path_item in picture_path_list:
            image_path = os.path.join(picture_path, picture_path_item)
            image_tensor = load_and_preprocess_image(tf.io.read_file(filename=image_path), data_augmentation=False)
            image_tensor_list.append(image_tensor)
    image_tensor_list_length = len(image_tensor_list)
    images = tf.stack(image_tensor_list, axis=0)
    prediction = model(images, training=False)
    pred_class_num_list = tf.math.argmax(prediction, axis=-1)
    pred_class_name_list = []
    true_num = 0
    for index, pred_class_num_item in enumerate(pred_class_num_list):
        if class_name_list[pred_class_num_item] == class_name_list[int(index//5)]:
            true_num = true_num + 1
        pred_class_name_item = class_name_list[pred_class_num_item]
        pred_class_name_list.append(pred_class_name_item)
    return pred_class_name_list, image_tensor_list_length, true_num


if __name__ == '__main__':
    # GPU settings
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    # load the model
    model_create_start_time = timer()
    model = get_model()
    # model.load_weights(filepath=save_model_dir+"model")
    # model.load_weights(filepath=save_model_dir+"model_index-"+'{}-epochs-{}'.format(model_index, EPOCHS))
    save_model_path = save_model_dir + '{}-epochs-{}-batch-{}/'.format(model_name_list[model_index], EPOCHS, BATCH_SIZE)
    model.load_weights(filepath=save_model_path+'{}-epochs-{}-batch-{}'.format(model_name_list[model_index], EPOCHS, BATCH_SIZE))
    model_create_end_time = timer()

    print('model create spend : {} seconds'.format(model_create_end_time - model_create_start_time))

    # pred_class = get_single_picture_prediction(model, test_image_dir)
    model_predict_start_time = timer()
    # pred_class_name, pred_num, pred_true_num = get_list_picture_prediction(model, test_image_dir)
    # acc = pred_true_num / pred_num
    pred_class_name = get_single_picture_prediction(model, 'test_dataset_single/')
    model_predict_end_time = timer()
    
    # print('every predict spend : {} seconds'.format((model_predict_end_time - model_predict_start_time) / pred_num))
    print('one predict spend : {} seconds'.format(model_predict_end_time - model_predict_start_time))

    # print('{} predict spend : {} seconds'.format(pred_num, model_predict_end_time - model_predict_start_time))
    print(pred_class_name)
    # print('the test acc is {}, the proportion {}/{}'.format(acc, pred_true_num, pred_num))
