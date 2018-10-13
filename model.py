import tensorflow as tf

def create_sub_conv(ind, inputs, filters):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name='sub_conv{}'.format(ind)
    )

def create_sub_pool(ind, inputs):
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=2,
        padding='valid',
        name='sub_pool{}'.format(ind)
    )

def create_mid_conv(ind, inputs, filters=128, kernel_size=7, activation=tf.nn.relu):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=[1, 1],
        padding='same',
        activation=activation,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name='mid_conv{}'.format(ind)
    )


class Model(object):
    def __init__(self):
        self.input_size = 256
        self.heatmap_size = 32
        self.joints = 21
        self.stages = 3
        self.stage_heatmaps = []
        self.stage_losses = []
        
        self.input_images = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.input_size, self.input_size, 3),
            name='input_images'
        )
        self.cmap_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.input_size, self.input_size, 1),
            name='cmap_placeholder'
        )
        self.gt_hmap_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.heatmap_size, self.heatmap_size, self.joints + 1),
            name='gt_hmap_placeholder'
        )
        
        self._build_cnn()

    def _build_cnn(self):
        scope_name = 'sub_stages'
        print('Building scope: ' + scope_name)
        with tf.variable_scope(scope_name):
            # 256 x 256 x 3
            sub_conv1 = create_sub_conv(1, self.input_images, 64)
            # 256 x 256 x 64
            sub_conv2 = create_sub_conv(2, sub_conv1, 64)
            # 256 x 256 x 64
            sub_pool1 = create_sub_pool(1, sub_conv2)
            # 128 x 128 x 64
            sub_conv3 = create_sub_conv(3, sub_pool1, 128)
            # 128 x 128 x 128
            sub_conv4 = create_sub_conv(4, sub_conv3, 128)
            # 128 x 128 x 128
            sub_pool2 = create_sub_pool(2, sub_conv4)
            # 64 x 64 x 128
            sub_conv5 = create_sub_conv(5, sub_pool2, 256)
            # 64 x 64 x 256
            sub_conv6 = create_sub_conv(6, sub_conv5, 256)
            # 64 x 64 x 256
            sub_conv7 = create_sub_conv(7, sub_conv6, 256)
            # 64 x 64 x 256
            sub_conv8 = create_sub_conv(8, sub_conv7, 256)
            # 64 x 64 x 256
            sub_pool3 = create_sub_pool(3, sub_conv8)
            # 32 x 32 x 256
            sub_conv9 = create_sub_conv(9, sub_pool3, 512)
            # 32 x 32 x 512
            sub_conv10 = create_sub_conv(10, sub_conv9, 512)
            # 32 x 32 x 512
            sub_conv11 = create_sub_conv(11, sub_conv10, 512)
            # 32 x 32 x 512
            sub_conv12 = create_sub_conv(12, sub_conv11, 512)
            # 32 x 32 x 512
            sub_conv13 = create_sub_conv(13, sub_conv12, 512)
            # 32 x 32 x 512
            sub_conv14 = create_sub_conv(14, sub_conv13, 512)
            # 32 x 32 x 512
            self.sub_stage_img_feature = tf.layers.conv2d(
                inputs=sub_conv14,
                filters=128,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='sub_stage_img_feature'
            )
            # 32 x 32 x 128

        scope_name = 'stage_1'
        print('Building scope: ' + scope_name)
        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(
                inputs=self.sub_stage_img_feature,
                filters=512,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='conv1'
            )
            # 32 x 32 x 512
            heatmap = tf.layers.conv2d(
                inputs=conv1,
                filters=self.joints+1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='valid',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='stage_heatmap'
            )
            # 32 x 32 x (21 + 1)
            self.stage_heatmaps.append(heatmap)

        self._build_stage(2)
        self._build_stage(3)
        
    def _build_stage(self, stage):
        scope_name = 'stage_{}'.format(stage)
        print('Building scope: ' + scope_name)
        with tf.variable_scope(scope_name):
            # 32 x 32 x 22
            # 32 x 32 x 512
            feature_map = tf.concat([
                self.stage_heatmaps[-1],
                self.sub_stage_img_feature
            ], axis=3)
            # 32 x 32 x 532
            mid_conv1 = create_mid_conv(1, feature_map)
            # 32 x 32 x 128
            mid_conv2 = create_mid_conv(2, mid_conv1)
            mid_conv3 = create_mid_conv(3, mid_conv2)
            mid_conv4 = create_mid_conv(4, mid_conv3)
            mid_conv5 = create_mid_conv(5, mid_conv4)
            mid_conv6 = create_mid_conv(6, mid_conv5, filters=128, kernel_size=1)
            # 32 x 32 x 128
            heatmap = create_mid_conv(7, mid_conv6, filters=self.joints+1,
                kernel_size=1, activation=None)
            # 32 x 32 x 22
            self.stage_heatmaps.append(heatmap)

    # def _build_loss(self):
    #     self.total_loss = 0
    #     self.total_loss_eval = 0
    #     self.init_lr = lr
    #     self.lr_decay_rate = lr_decay_rate
    #     self.lr_decay_step = lr_decay_step
    #     self.optimizer = 'Adam'
    #     self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

    #     for i in range(self.stages):
    #         stage = i + 1
    #         scope_name = 'stage{}_loss'.format(stage)
    #         print('Building loss:', scope_name)
    #         with tf.variable_scope(scope_name):
    #             loss = tf.nn.l2_loss(
    #                 self.stage_heatmaps[i] - self.gt_hmap_placeholder,
    #                 name='l2_loss'
    #             ) / self.batch_size
    #             self.stage_losses.append()


