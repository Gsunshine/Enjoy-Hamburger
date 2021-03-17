from generator import generator as gen
from generator import make_z_normal
import tensorflow as tf
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from categories import indx2category


flags.DEFINE_integer('bs', 8, 'Batch size for image generation.')
flags.DEFINE_string('category', 'valley', 'Category of generated images.'
                    'See file: categories.py for all available categories.')
flags.DEFINE_integer('seed', 10, 'Seed for numpy.')
flags.DEFINE_integer('num_bs', 10, 'Total number of generated batches.')
flags.DEFINE_string('pretrained_path',
                    'model.ckpt-900000',
                    'Folder in which the pretrained model is located.')
flags.DEFINE_string('out_dir', 'samples/', 'Output directory for images')


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
    category2indx = {v:k for k, v in indx2category.items()}
    bs = flags.FLAGS.bs
    num_batches = flags.FLAGS.num_bs
    name = flags.FLAGS.category
    category = category2indx[name]

    # create target category
    target = tf.fill([bs], category)

    with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
        # run, for variables creation
        latent = tf.random.truncated_normal(shape=(bs, 128))
        feed(latent, target, training=True)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
            # run build
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            # run restore
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, flags.FLAGS.pretrained_path)

            with tf.device('/GPU:0'):
                latent = tf.random.truncated_normal(shape=(bs, 128))
                img, _ = feed(latent, target)

            indx = 0
            for i in range(num_batches):
                sess.run(img)
                for j in range(bs):
                    indx += 1
                    plot_img = change_range(img[j].eval(), -1, 1, 0, 255).astype(np.uint8)
                    save_img_cv2(plot_img, os.path.join(flags.FLAGS.out_dir, f'{name}_{indx}.png'))

if __name__ == '__main__':
    app.run(main)
