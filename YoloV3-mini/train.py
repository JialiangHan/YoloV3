#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
# ================================================================
from YoloV3_mini import yolov3_mini
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import decode, compute_loss
from core.config import cfg

gpus = tf.config.list_physical_devices(device_type="GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
trainset = Dataset('train')
logdir = "./data/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors1 = yolov3_mini(input_tensor, 3 * (NUM_CLASS + 5))

output_tensors1 = []
for i, conv_tensor in enumerate(conv_tensors1):
    pred_tensor = decode(conv_tensor, i)
    output_tensors1.append(conv_tensor)
    output_tensors1.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors1)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

if cfg.load_weights_before_training:
    model.load_weights(filepath=cfg.save_model_dir + "epoch-{}".format(cfg.load_weights_from_epoch))
    print("Successfully load weights!")
else:
    load_weights_from_epoch = -1


def train_step(epoch, image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> Epoch:%4d/%4d STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (epoch, cfg.TRAIN.EPOCHS, global_steps, optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


for epoch in range(cfg.load_weights_from_epoch + 1, cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        train_step(epoch, image_data, target)
    # model.save_weights("./yolov3_mini/")
    model.save_weights(filepath=cfg.save_model_dir + "epoch-{}".format(epoch), save_format='tf')
model.save_weights(filepath=cfg.save_model_dir + "saved_model", save_format='tf')
