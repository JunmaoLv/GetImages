import cv2
import tensorflow as tf
from predict import get_single_picture_prediction_from_camera
from prepare_data import load_and_prepocess_image_from_camera
from configuration import save_model_dir, test_image_dir, model_index, EPOCHS, model_name_list, BATCH_SIZE
from train import get_model
import os
from timeit import default_timer as timer


# load the model
model_create_start_time = timer()
model = get_model()

save_model_path = save_model_dir + '{}-epochs-{}-batch-{}/'.format(model_name_list[model_index], EPOCHS, BATCH_SIZE)
model.load_weights(filepath=save_model_path+'{}-epochs-{}-batch-{}'.format(model_name_list[model_index], EPOCHS, BATCH_SIZE))
model_create_end_time = timer()

print('model create spend : {} seconds'.format(model_create_end_time - model_create_start_time))

cap = cv2.VideoCapture(0)
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

picture_saved_path = 'saved_images/'
picture_saved_num = 1
pred_num = 1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    # print('the shape of frame: {}'.format(frame.shape))
    input = cv2.waitKey(20)
    if input==ord('q'):
        break
    elif input==ord('s'):
        picture_saved_name = picture_saved_path+'image-{}.jpg'.format(picture_saved_num)
        print('saving frame to {}'.format(picture_saved_name))
        cv2.imwrite(picture_saved_name, frame)
        print('saving {} frame finished'.format(picture_saved_num))
        picture_saved_num = picture_saved_num + 1
    elif input==ord('p'):    
        model_predict_start_time = timer()
        image = load_and_prepocess_image_from_camera(frame)
        pred_class_name = get_single_picture_prediction_from_camera(model, image)
        model_predict_end_time = timer()
        print('{}th predict spend : {} seconds'.format(pred_num, model_predict_end_time - model_predict_start_time))
        print('{}th prediction result: {}'.format(pred_num, pred_class_name))
        
        cv2.putText(frame, pred_class_name, (40, 50), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), 2)
        cv2.imshow(pred_class_name, frame)
        cv2.waitKey(1000)
        pred_num = pred_num + 1

cap.release()
cv2.destroyAllWindows()
