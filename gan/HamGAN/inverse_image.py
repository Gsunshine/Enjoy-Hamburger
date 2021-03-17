from generator import generator as gen
from discriminator import discriminator as disc
import tensorflow as tf
from absl import app, flags
import numpy as np
from optimization import BaseLookAhead
import cv2
import matplotlib.pyplot as plt
from categories import indx2category
import os

flags.DEFINE_integer('bs', 8, 'Number of tries for a single image inversion.')
flags.DEFINE_string('category', 'maltese', 'Category to run inversion on.')
flags.DEFINE_integer('seed', 10, 'Seed for numpy.')
flags.DEFINE_integer('steps', 1000, 'Inversion steps')
flags.DEFINE_float('lr', 0.07, 'Learning rate for Adam optimizer')
flags.DEFINE_float('std', 0.2, 'Noise std.')
flags.DEFINE_string('image_path', 'real_images/maltese.jpeg',
                    'Image to peform inversion on. '
                    'Leave empty for self inversion')
flags.DEFINE_boolean('inject_noise', False,
                     'Whether to inject noise to image.')
flags.DEFINE_string('pretrained_path',
                    'model.ckpt-900000',
                    'Folder in which the pretrained model is located.')
flags.DEFINE_string('out_dir', 'inversions/',
                    'Directory for output files.')


def save_img_cv2(img, name):
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(name, img_color)


def feed(x, target, training=False):
    return gen(x, target, 64, 1000, training=training)


def feed_disc(img, target):
    return disc(img, target, 64, 1000)[:3]


def discriminator_loss(loss_vars):
    latent_, img, attn, attn_map, target, step = loss_vars
    with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
        img_ = feed(latent_, target)[0]
    with tf.compat.v1.variable_scope('Discriminator', reuse=tf.compat.v1.AUTO_REUSE):
        attn_, attn_map_ = feed_disc(img_, target)[1:3]
    img_loss = tf.norm(img - img_)
    disc_loss = 0
    heads = [0, 1, 2, 3]
    for head in heads:
        disc_loss += weighted_norm(attn, attn_, attn_map, (32, 32), head=head)
    return disc_loss, img_loss


def get_saliency(attn_map, shape=(128, 128), method='nearest'):
    bs = attn_map.shape[0]
    attn_map = tf.reshape(attn_map, [bs, -1, 256])
    non_zeros = tf.math.count_nonzero(attn_map, axis=1)
    weights = tf.math.reduce_sum(attn_map, axis=1) / tf.cast(non_zeros, tf.float32)
    weights = tf.reshape(weights, [bs, 16, 16, 1])
    weights = tf.image.resize(weights, shape, method=method)
    return weights


def weighted_norm(x, x_, attn_map, shape, head=None, order=2, method='nearest'):
    if head is not None:
        attn_map = attn_map[:, head]
    n_weights = get_saliency(attn_map, shape, method=method)
    w_diff = tf.math.multiply(x - x_, n_weights)
    loss_value = tf.norm(w_diff, ord=order)
    return loss_value


def change_range(x, a, b, c, d):
    ''' Maps value x from range [a, b] to range [c, d] '''
    return (d - c) / (b - a) * (x - a) + c


def visualize_attention(img, attn_map, head=None,
                        shape=(128, 128), method='bilinear',
                        scatter_points=None,
                        cmap='magma',
                        img_opacity=0.35):
    ''' Optionally call this function to generate saliencies '''

    sal_path = os.path.join(flags.FLAGS.out_dir, 'saliencies/')
    tf.io.gfile.mkdir(sal_path)
    # visualize saliency
    fig = plt.figure()
    plt.axis('off')
    saliencies = np.squeeze(get_saliency(attn_map, method=method, shape=shape).eval())
    plt.imshow(np.average(saliencies, axis=0), alpha=1 - img_opacity, cmap=cmap)
    plt.imshow(img, alpha=img_opacity)
    plt.savefig(sal_path + 'saliency.png', bbox_inches='tight')

    # visualize per head saliency
    for i in range(8):
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(saliencies[i], alpha=1 - img_opacity, cmap=cmap)
        plt.imshow(img, alpha=img_opacity)
        plt.savefig(sal_path + f'saliency_{i}.png', bbox_inches='tight')


def main(_):
    # create output dir if it does not exist
    tf.io.gfile.mkdir(flags.FLAGS.out_dir)
    ''' Main function '''
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    category2indx = {v:k for k, v in indx2category.items()}

    name = flags.FLAGS.category
    indx = category2indx[name]

    np.random.seed(flags.FLAGS.seed)
    tf.compat.v2.random.set_seed(flags.FLAGS.seed)
    latent = np.random.normal(size=(flags.FLAGS.bs, 128)).astype(np.float32)

    target = tf.fill([flags.FLAGS.bs], indx)

    with tf.device('/GPU:0'):
        opt = tf.compat.v1.train.AdamOptimizer(flags.FLAGS.lr)

    with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
        img, _, = feed(latent, target, training=True)

    with tf.compat.v1.variable_scope('Discriminator', reuse=tf.compat.v1.AUTO_REUSE):
        feed_disc(img, target)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, flags.FLAGS.pretrained_path)

        if not flags.FLAGS.image_path:
            with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
                img, g_attn_map = feed(latent, target)
                if flags.FLAGS.inject_noise:
                    img += tf.random.normal([flags.FLAGS.bs, 128, 128, 3], stddev=flags.FLAGS.std, mean=0)
        else:
            img = change_range(plt.imread(flags.FLAGS.image_path), 0, 255, -1, 1).astype(np.float32)
            img = tf.image.resize(img, (128, 128)).eval()
            img = np.expand_dims(img, axis=0)
            if flags.FLAGS.inject_noise:
                img += tf.random.normal([flags.FLAGS.bs, 128, 128, 3], stddev=flags.FLAGS.std, mean=0)
                img = img.eval()
            img = np.repeat(img, flags.FLAGS.bs, axis=0)

        with tf.compat.v1.variable_scope('Discriminator', reuse=tf.compat.v1.AUTO_REUSE):
            _, d_attn, d_attn_map = feed_disc(img, target)


        step = tf.Variable(0, trainable=False)
        latent_ = tf.Variable(tf.random.truncated_normal(shape=(flags.FLAGS.bs, 128)), trainable=True, name='latent_')
        sess.run(tf.compat.v1.variables_initializer([latent_, step]))

        lookahead = BaseLookAhead([latent_], k=5, alpha=0.5)
        sess.run(lookahead.get_ops())

        loss_vars = [latent_, img, d_attn, d_attn_map, target, step]
        loss_value, img_loss = discriminator_loss(loss_vars)

        update_op = opt.minimize(loss_value, var_list=[latent_])
        sess.run(tf.compat.v1.variables_initializer(opt.variables()))

        for i in range(flags.FLAGS.steps):
            print('{0}/{1}:\t Loss: {2:.4f}\t Image loss: {3:.4f}'.format(
                i, flags.FLAGS.steps, loss_value.eval(), img_loss.eval()))
            sess.run(update_op)

        with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
            inverse, g_attn_map_ = feed(latent_, target)
            inverse = inverse.eval()
            g_attn_map_ = g_attn_map_.eval()
            inverse = change_range(inverse, -1, 1, 0, 255).astype(np.uint8)


        if not flags.FLAGS.image_path:
            img = img.eval()

        # Uncomment to visualize saliencies
        # plot_img = change_range(img, -1, 1, 0, 255).astype(np.uint8)[0]
        # visualize_attention(plot_img, g_attn_map_[0])


        img = change_range(img, -1, 1, 0, 255).astype(np.uint8)
    save_img_cv2(img[0], os.path.join(flags.FLAGS.out_dir,
                                      f'real_{flags.FLAGS.seed}.png'))
    for i in range(flags.FLAGS.bs):
        save_img_cv2(inverse[i], os.path.join(flags.FLAGS.out_dir, f'reconstructed_{flags.FLAGS.seed + i}.png'))


if __name__ == '__main__':
    app.run(main)
