# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import torchfile
import matplotlib.pyplot as plt

# 拼接图片 横1 纵0
def connect(images, axis):
    # for i in range(len(images)):
    np.concatenate(images,axis)

# 缩放图片
# 将放得过大的图缩回RGB图像
def any_to_unit8_scale(image):
    float_image = image.astype(np.float32)
    imax = float_image.max()
    imin = float_image.min()
    diff = abs(imax - imin)
    # 将图片归一化
    normalized_image = (float_image - imin) / diff
    # 返回原图
    return (normalized_image * 255).astype(np.uint8)

# 修整图片
# 可以将图片中的小数都变成整数
def any_to_uint8_clip(image):
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# 从placeholder中载入图片
def image_from_file(graph, placeholder_name, size=None):
    # 可以依次处理多张输入图片
    with graph.as_default():
        filename = tf.placeholder(tf.string, name=placeholder_name)
        image = tf.image.decode_jpeg(tf.read_file(filename))
        image = tf.expand_dims(image, 0)
        image = preprocess_image(image, size)
        return image, filename

# 预处理图片
def preprocess_image(image, size=None):
    image = tf.reverse(image, axis=[-1])
    # 将图片处理为0~1之间
    image = tf.cast(image, tf.float32) / 256.0
    # 如果设置图片大小，将其变化为设置大小
    if size is not None:
        image = tf.image.resize_images(image, size)
    return image

# 将图片转回RGB
def postprocess_image(image, size=None):
    image = image * 256
    image = tf.reverse(image, axis=[-1])
    return image

# 
def show_image(image):
    # fig = plt.figure()
    plt.imshow(image)
    plt.show()

# 自适应实例标准化算法
def AdaIN(content_features, style_features, alpha):
    # tf.nn.moments 返回值mean表示均值（一阶） variance表示方差（中心矩阵）
    style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
    epsilon = 1e-5
    # 批量标准化
    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                            content_variance, style_mean,
                                                            tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features
    
# 载入torch网络
def graph_from_t7(net, graph, t7_file):
    layers = []
    print_layers = []  # [0, 30]
    # 导入训练好的网络，Windows下用命令行执行会出现缓冲区正在载入错误
    t7 = torchfile.load(t7_file, force_8bytes_long=True)

    with graph.as_default():

        for idx, module in enumerate(t7.modules):

            if idx in print_layers:
                print(module)

            if module._typename == b'nn.SpatialReflectionPadding':
                left = module.pad_l
                right = module.pad_r
                top = module.pad_t
                bottom = module.pad_b
                net = tf.pad(net, [[0, 0], [top, bottom], [left, right], [0, 0]], 'REFLECT')
                layers.append(net)
            elif module._typename == b'nn.SpatialConvolution':
                weight = module.weight.transpose([2, 3, 1, 0])
                bias = module.bias
                strides = [1, module.dH, module.dW, 1]  # Assumes 'NHWC'
                net = tf.nn.conv2d(net, weight, strides, padding='VALID')
                net = tf.nn.bias_add(net, bias)
                layers.append(net)
            # ReLU [-1,0] [0,0] [1,1] 激活函数
            elif module._typename == b'nn.ReLU':
                net = tf.nn.relu(net)
                layers.append(net)
            elif module._typename == b'nn.SpatialUpSamplingNearest':
                d = tf.shape(net)
                size = [d[1] * module.scale_factor, d[2] * module.scale_factor]
                net = tf.image.resize_nearest_neighbor(net, size)
                layers.append(net)
            elif module._typename == b'nn.SpatialMaxPooling':
                net = tf.nn.max_pool(net, ksize=[1, module.kH, module.kW, 1], strides=[1, module.dH, module.dW, 1],
                                     padding='VALID', name=str(module.name, 'utf-8'))
                layers.append(net)
            else:
                raise NotImplementedError(module._typename)

        return net, layers

# 风格化函数
def stylize(content, style, vgg_t7_file, decode_t7_file, alpha=0.5, resize=None):
    print(content, style, vgg_t7_file, decode_t7_file, alpha)
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess, tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        c, c_filename = image_from_file(g, 'content_image', size=resize)
        s, s_filename = image_from_file(g, 'style_image', size=resize)
        print(vgg_t7_file)
        _, c_vgg = graph_from_t7(c, g, vgg_t7_file)
        _, s_vgg = graph_from_t7(s, g, vgg_t7_file)
        c_vgg = c_vgg[30]
        s_vgg = s_vgg[30]
        stylized_content = AdaIN(c_vgg, s_vgg, alpha)
        c_decoded, _ = graph_from_t7(stylized_content, g, decode_t7_file)
        c_decoded = postprocess_image(c_decoded)
        c = postprocess_image(c)
        s = postprocess_image(s)
        feed_dict = {c_filename: content, s_filename: style}
        combined, style_image, content_image = sess.run([c_decoded, s, c], feed_dict=feed_dict)
        return np.squeeze(combined), np.squeeze(content_image), np.squeeze(style_image)