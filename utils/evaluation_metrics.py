"""
Evaluation metrics
"""

import numpy as np
import sklearn.metrics as metrics
import os
import glob
import cv2
from PIL import Image


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 1) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def multilabel_numeric(pred, target):
    cm = metrics.confusion_matrix(target, pred, labels=np.arange(np.max(target) + 1))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (TP + FN + FP)
    return TP, FP, TN, FN


def get_class_weights(pred, taret):
    weights = np.bincount(taret) / len(taret)
    return weights


def wAcc(pred, target):
    weights = get_class_weights(pred, target)
    TP, FP, TN, FN = multilabel_numeric(pred, target)
    accuracy = 0
    for i in range(np.max(target) + 1):
        acc = weights[i] * ((TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]))
        accuracy += acc
    # print(r"wAcc: {0:.4f}".format(accuracy))
    level_acc = []
    for i in range(np.max(target) + 1):
        acc = ((TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]))
        level_acc.append(acc)

    return [accuracy, level_acc]


def wSe(pred, target):
    weights = get_class_weights(pred, target)
    TP, FP, TN, FN = multilabel_numeric(pred, target)
    senitivity = 0
    for i in range(np.max(target) + 1):
        sen = weights[i] * (TP[i] / (TP[i] + FN[i]))
        senitivity += sen
    # print(r"wSe: {0:.4f}".format(senitivity))
    level_sen = []
    for i in range(np.max(target) + 1):
        sen = (TP[i] / (TP[i] + FN[i]))
        level_sen.append(sen)

    return [senitivity, level_sen]


def wSp(pred, target):
    weights = get_class_weights(pred, target)
    TP, FP, TN, FN = multilabel_numeric(pred, target)
    specificity = 0
    for i in range(np.max(target) + 1):
        spe = weights[i] * (TN[i] / (TN[i] + FP[i]))
        specificity += spe
    # print(r"wSp: {0:.4f}".format(specificity))
    level_spe = []
    for i in range(np.max(target) + 1):
        spe = (TN[i] / (TN[i] + FP[i]))
        level_spe.append(spe)

    return [specificity, level_spe]


def get_metrix(pred, target):
    accuracy = wAcc(pred, target)
    sensitivity = wSe(pred, target)
    specificity = wSp(pred, target)
    return accuracy, sensitivity, specificity
