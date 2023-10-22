from pathlib import Path

import scipy.io as sio
import numpy as np
import cv2


#########################
def process_face_grid(face_grid_2d, face_grid_size):
    # Create grid (e.g. [25 x 25])
    face_grid_2d = face_grid_2d.T
    face_grid_2d = face_grid_2d * face_grid_size
    face_grid_2d = face_grid_2d.astype(int)

    x_max, y_max = np.amax(face_grid_2d, axis=1)
    x_min, y_min = np.amin(face_grid_2d, axis=1)

    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def load_screen_sizes(base_data_path):
    screen_size = sio.loadmat(f'{base_data_path}Calibration/screenSize.mat')

    sizes = (screen_size['width_pixel'][0][0],
             screen_size['height_pixel'][0][0],
             screen_size['width_mm'][0][0],
             screen_size['height_mm'][0][0])

    return sizes


def print_normalization_stats(correct_total, errors_total):
    print(f'\nTotal correctly processed images: {correct_total}')
    print(f'Total images without face / landmarks detected: {errors_total}')

    error_percentage = (errors_total / (errors_total + correct_total)) * 100
    print(f'Percentual dropout rate: {error_percentage}\n\n')


def save_normalized(image_metadata, base_processed_path, participant, eyes_together, processed_eyes, face_grid,
                    screen_sizes, eye_screen_distance):
    day_directory_name = image_metadata[0].split('/')[0]
    file_name_no_suffix = image_metadata[0].split('/')[1].split('.')[0]

    # Create directories if not exists
    Path(base_processed_path + participant + '/' + day_directory_name).mkdir(parents=True, exist_ok=True)

    path_prefix = base_processed_path + participant + '/' + day_directory_name + '/'

    # If eyes processed together - save only 1 file
    if eyes_together:
        cv2.imwrite(f'{path_prefix}{file_name_no_suffix}_eyes.jpg', processed_eyes)

        return [f'{day_directory_name}/{file_name_no_suffix}_eyes.jpg', image_metadata[1], image_metadata[2],
                screen_sizes[0], screen_sizes[1], screen_sizes[2], screen_sizes[3],
                face_grid['x'], face_grid['y'], face_grid['width'], face_grid['height'],
                eye_screen_distance]

    # If eyes processed separately - save 2 files
    else:
        cv2.imwrite(f'{path_prefix}{file_name_no_suffix}_right_eye.jpg', processed_eyes[0])
        cv2.imwrite(f'{path_prefix}{file_name_no_suffix}_left_eye.jpg', processed_eyes[1])

        return [f'{day_directory_name}/{file_name_no_suffix}_right_eye.jpg',
                f'{day_directory_name}/{file_name_no_suffix}_left_eye.jpg',
                image_metadata[1], image_metadata[2],
                screen_sizes[0], screen_sizes[1], screen_sizes[2], screen_sizes[3],
                face_grid['x'], face_grid['y'], face_grid['width'], face_grid['height'],
                eye_screen_distance]
