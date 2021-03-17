"""
In this file, we implement NMF and CD for the Hamburger module.

Refer to the LICENCE of the tensorflow-gan library for using parts of the
code we did not change.
"""
import tensorflow as tf

# Use opt_einsum to further accelerate einsum.
# from opt_einsum import contract

contract = tf.einsum


def nmf(x, shape, steps=6):
    print('Add NMF!')
    x = tf.nn.relu(x)

    bases = tf.random.uniform(shape, name='init_bases')

    coef = contract('bdn,bdr->bnr', x, bases)
    coef = tf.nn.softmax(coef)

    for now_iter in range(steps):
        with tf.compat.v1.variable_scope('iter_' + str(now_iter)):
            numerator = contract('bdn,bdr->bnr', x, bases)
            denominator = contract('bnk,bdk,bdr->bnr', coef, bases, bases)
            coef = coef * numerator / (denominator + tf.constant(1e-6))

            numerator = contract('bdn,bnr->bdr', x, coef)
            denominator = contract('bdk,bnk,bnr->bdr', bases, coef, coef)
            bases = bases * numerator / (denominator + tf.constant(1e-6))

    coef, bases = tf.stop_gradient(coef), tf.stop_gradient(bases)

    numerator = contract('bdn,bdr->bnr', x, bases)
    denominator = contract('bnk,bdk,bdr->bnr', coef, bases, bases)
    coef = coef * numerator / (denominator + tf.constant(1e-6))

    x = contract('bnr,bdr->bdn', coef, bases)

    return x


def cd(x, shape, steps=6):
    print('Add CD!')
    bases = tf.random.normal(shape, name='init_bases')
    bases = tf.math.l2_normalize(bases, axis=1)
    
    std_x = tf.math.l2_normalize(x, axis=1)

    for now_iter in range(steps):
        with tf.compat.v1.variable_scope('iter_' + str(now_iter)):
            coef = contract('bdn,bdr->bnr', std_x, bases)
            coef = tf.nn.softmax(100 * coef, axis=-1)
            coef = coef /(1e-6 + tf.math.reduce_sum(coef, axis=1, keepdims=True))
            
            bases = contract('bdn,bnr->bdr', x, coef)
            bases = tf.math.l2_normalize(bases, axis=1)
    
    coef, bases = tf.stop_gradient(coef), tf.stop_gradient(bases)
    
    temp = contract('bdr,bdk->brk', bases, bases) + 0.01 * tf.eye(shape[-1])
    temp = tf.linalg.inv(temp)
    coef = contract('bdn,bdr,brk->bnk', x, bases, temp)

    x = contract('bnr,bdr->bdn', coef, bases)

    return x


def get_ham(key):
    hams = {'NMF': nmf,
            'CD': cd}

    assert key in hams

    return hams[key]
