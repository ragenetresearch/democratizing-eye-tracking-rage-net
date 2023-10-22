import tensorflow as tf


def retrieve_and_normalize_image(file, efficientnet=False, img_size=(60, 36)):
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size)

    if efficientnet:
        img = (img - 128.00) / 128.00
    else:
        img = img / 255

    return img


def read_images_separate(right_eye_img, left_eye_img, label, efficientnet=False, img_size=(60, 36)):
    right_eye_img = retrieve_and_normalize_image(right_eye_img, efficientnet=efficientnet, img_size=img_size)
    left_eye_img = retrieve_and_normalize_image(left_eye_img, efficientnet=efficientnet, img_size=img_size)

    return (right_eye_img, left_eye_img), label
