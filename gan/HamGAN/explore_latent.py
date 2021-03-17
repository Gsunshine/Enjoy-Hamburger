from generator import generator as gen
from generator import make_z_normal
import tensorflow as tf
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from categories import indx2category
from scipy.stats import truncnorm

# Random states: 4, 99, 3, 900, 412, 109
flags.DEFINE_string('category', 'maltese', 'Category of generated images.'
                    'See file: categories.py for all available categories.')
flags.DEFINE_integer('seed', 10, 'Seed for numpy.')
flags.DEFINE_string('pretrained_path',
                    'model.ckpt-900000',
                    'Folder in which the pretrained model is located.')
flags.DEFINE_string('out_dir', 'interpolations/',
                    'Output directory for images')


def interpolate_points(p1, p2, n_steps=20):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


def save_img_cv2(img, name):
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(name, img_color)


def feed(x, target, training=False):
    return gen(x, target, 64, 1000, training=training)[:2]


def change_range(x, a, b, c, d):
    ''' Maps value x from range [a, b] to range [c, d] '''
    return (d - c) / (b - a) * (x - a) + c


def main(_):
    # create out_dir
    tf.io.gfile.mkdir(flags.FLAGS.out_dir)

    # disable eager execution
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    # restore cli parameters
    category2indx = {v: k for k, v in indx2category.items()}
    name = flags.FLAGS.category
    category = category2indx[name]

    # create target category
    target = tf.fill([1], category)


    with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
        # run, for variables creation
        latent = tf.random.truncated_normal(shape=(1, 128))
        feed(latent, target, training=True)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
            # run build
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            # run restore
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, flags.FLAGS.pretrained_path)

            latent_1 = truncnorm.rvs(-1, 1, size=(1, 128)).astype(np.float32)
            latent_2 = truncnorm.rvs(-1, 1, size=(1, 128)).astype(np.float32)
            latent_vars = interpolate_points(latent_1, latent_2)

            img, _ = feed(latent_1, target)
            plot_img = change_range(img[0].eval(), -1, 1, 0, 255).astype(np.uint8)
            save_img_cv2(plot_img, os.path.join(flags.FLAGS.out_dir, 'latent_1.png'))

            img, _ = feed(latent_2, target)
            plot_img = change_range(img[0].eval(), -1, 1, 0, 255).astype(np.uint8)
            save_img_cv2(plot_img, os.path.join(flags.FLAGS.out_dir, 'latent_2.png'))

            for i, latent in enumerate(latent_vars):
                img, _ = feed(latent, target)
                plot_img = change_range(img[0].eval(), -1, 1, 0, 255).astype(np.uint8)
                save_img_cv2(plot_img, os.path.join(flags.FLAGS.out_dir, f'{i}.png'))


if __name__ == '__main__':
    app.run(main)
