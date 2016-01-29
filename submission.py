from __future__ import print_function

import csv
import numpy as np

from model import get_model
from utils import real_to_cdf, preprocess


def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load('data/X_validate.npy')
    ids = np.load('data/ids_validate.npy')

    X = X.astype(np.float32)
    X /= 255

    return X, ids


def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


def submission():
    """
    Generate submission file for the trained models.
    """
    print('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    print('Loading models weights...')
    model_systole.load_weights('weights_systole_best.hdf5')
    model_diastole.load_weights('weights_diastole_best.hdf5')

    # load val losses to use as sigmas for CDF
    with open('val_loss.txt', mode='r') as f:
        val_loss_systole = float(f.readline())
        val_loss_diastole = float(f.readline())

    print('Loading validation data...')
    X, ids = load_validation_data()

    print('Pre-processing images...')
    X = preprocess(X)

    batch_size = 32
    print('Predicting on validation data...')
    pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
    pred_diastole = model_diastole.predict(X, batch_size=batch_size, verbose=1)

    # real predictions to CDF
    cdf_pred_systole = real_to_cdf(pred_systole, val_loss_systole)
    cdf_pred_diastole = real_to_cdf(pred_diastole, val_loss_diastole)

    print('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

    # write to submission file
    print('Writing submission to file...')
    fi = csv.reader(open('data/sample_submission_validate.csv'))
    f = open('submission.csv', 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0]))
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

    print('Done.')

submission()
