"""
        Definitions of generator functions.
Code in a slightly modified version of the SAGAN generator of the
tensorlow-gan library. We change only the part of the code which refers to
the attention layer. We do not claim any ownership on this code and you should
refer to the LICENCE of the tensorflow-gan library.
"""
from absl import flags
import tensorflow as tf
import tensorflow_gan as tfgan
import ops


def make_z_normal(num_batches, batch_size, z_dim):
    """Make random noise tensors with normal distribution.

    Args:
      num_batches: copies of batches
      batch_size: the batch_size for z
      z_dim: The dimension of the z (noise) vector.
    Returns:
      zs:  noise tensors.
    """
    shape = [num_batches, batch_size, z_dim]
    z = tf.random.normal(shape, name='z0', dtype=tf.float32)
    return z


def make_class_labels(batch_size, num_classes):
    """Generate class labels for generation."""
    # Uniform distribution.
    # TODO(augustusodena) Use true distribution of ImageNet classses.
    gen_class_logits = tf.zeros((batch_size, num_classes))
    gen_class_ints = tf.random.categorical(
        logits=gen_class_logits, num_samples=1)
    gen_class_ints.shape.assert_has_rank(2)
    gen_class_ints = tf.squeeze(gen_class_ints, axis=1)
    gen_class_ints.shape.assert_has_rank(1)

    return gen_class_ints


def usample(x):
    """Upsamples the input volume.

    Args:
      x: The 4D input tensor.
    Returns:
      An upsampled version of the input tensor.
    """
    # Allow the batch dimension to be unknown at graph build time.
    _, image_height, image_width, n_channels = x.shape.as_list()
    # Add extra degenerate dimension after the dimensions corresponding to the
    # rows and columns.
    expanded_x = tf.expand_dims(tf.expand_dims(x, axis=2), axis=4)
    # Duplicate data in the expanded dimensions.
    after_tile = tf.tile(expanded_x, [1, 1, 2, 1, 2, 1])
    return tf.reshape(after_tile,
                      [-1, image_height * 2, image_width * 2, n_channels])


def block(x, labels, out_channels, num_classes, name, training=True):
    """Builds the residual blocks used in the generator.

    Args:
      x: The 4D input tensor.
      labels: The labels of the class we seek to sample from.
      out_channels: Integer number of features in the output layer.
      num_classes: Integer number of classes in the labels.
      name: The variable scope name for the block.
      training: Whether this block is for training or not.
    Returns:
      A `Tensor` representing the output of the operation.
    """
    with tf.compat.v1.variable_scope(name):
        labels_onehot = tf.one_hot(labels, num_classes)
        x_0 = x
        x = tf.nn.relu(tfgan.tpu.batch_norm(x, training, labels_onehot,
                                            name='cbn_0'))
        x = usample(x)
        x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv1')
        x = tf.nn.relu(tfgan.tpu.batch_norm(x, training, labels_onehot,
                                            name='cbn_1'))
        x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv2')

        x_0 = usample(x_0)
        x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, training, 'snconv3')

        return x_0 + x


def generator(zs, target_class, gf_dim, num_classes, training=True):
    """Builds the generator segment of the graph, going from z -> G(z).

    Args:
      zs: Tensor representing the latent variables.
      target_class: The class from which we seek to sample.
      gf_dim: The gf dimension.
      num_classes: Number of classes in the labels.
      training: Whether in train mode or not. This affects things like batch
        normalization and spectral normalization.

    Returns:
      - The output layer of the generator.
      - A list containing all trainable varaibles defined by the model.
    """
    with tf.compat.v1.variable_scope(
            'generator', reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:

        act0 = ops.snlinear(
            zs, gf_dim * 16 * 4 * 4, training=training, name='g_snh0')
        act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

        # pylint: disable=line-too-long
        act1 = block(
            act0,
            target_class,
            gf_dim * 16,
            num_classes,
            'g_block1',
            training)  # 8
        act2 = block(
            act1,
            target_class,
            gf_dim * 8,
            num_classes,
            'g_block2',
            training)  # 16
        act3 = block(
            act2,
            target_class,
            gf_dim * 4,
            num_classes,
            'g_block3',
            training)  # 32
        if flags.FLAGS.G_module == 'hamburger':
            if flags.FLAGS.G_version == 'v1':
                print('Add V1 to G!')
                hamburger = ops.sn_hamburger_v1
            else:
                print('Add V2 to G!')
                hamburger = ops.sn_hamburger_v2
            act3 = hamburger(act3, target_class, num_classes,
                             training=training, name='g_ops',
                             ham_type=flags.FLAGS.G_ham_type,
                             S=flags.FLAGS.G_s,
                             D=flags.FLAGS.G_d,
                             R=flags.FLAGS.G_r,
                             steps=flags.FLAGS.G_K)
        else:
            print('No context module in G!')

        act4 = block(
            act3,
            target_class,
            gf_dim * 2,
            num_classes,
            'g_block4',
            training)  # 64
        act5 = block(
            act4,
            target_class,
            gf_dim,
            num_classes,
            'g_block5',
            training)  # 128
        act5 = tf.nn.relu(
            tfgan.tpu.batch_norm(
                act5,
                training,
                conditional_class_labels=None,
                name='g_bn'))
        act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, training, 'g_snconv_last')
        out = tf.nn.tanh(act6)
    var_list = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, gen_scope.name)
    return out, var_list
