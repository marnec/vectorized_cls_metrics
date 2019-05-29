import numpy as np

def binary_clf_curve(y_true, y_score):
    pos_label = 1.0

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    # desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    # desc_score_indices = np.unravel_index(np.argsort(y_score, axis=None, kind="mergesort")[::-1], y_score.shape)
    desc_score_indices = np.argsort(y_score, axis=0, kind="mergesort")[::-1]
    # print(desc_score_indices)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]


    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    # accumulate the true positives with decreasing threshold
    fps = 1 + threshold_idxs - tps

    return fps, tps, y_score[threshold_idxs]

def roc(fps, tps, thresholds):
    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def pr(fps, tps, thresholds):
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def confmat(fps, tps):
    # true negatives are given by
    tns = fps[-1] - fps
    # false negatives are given by
    fns = tps[-1] - tps
    # tn, fp, fn, tp
    a = np.array([tns, fps, fns, tps]).astype(int)
    return a

if __name__ == "__main__":
    size = 1000000
    nt = 10000
    t = np.greater_equal(np.random.rand(size), 0.5).astype(int)
    tt = t.reshape(size // nt, nt)
    # print(tt.shape)
    p = np.random.rand(size)
    pt = np.random.rand(size // nt, nt)

    # for x, y in zip(tt, pt):
    for i in range(size // nt):
        x = tt[i]
        y = pt[i]
        fps, tps, thr = binary_clf_curve(x, y)
    
    # print(np.random.rand(2, 4))

    # np.apply_along_axis(binary_clf_curve, 0, tt, pt)
    # fpr, tpr, thr = roc(fps, tps, thr)
    # pre, rec, thr = pr(fps, tps, thr)
    # print(confmat(fps, tps).T.shape)
    
