import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Tuple, List, Union

# relative imports
from parsers import parse_reference, parse_prediction
from logger import set_logger


def ignore_numpy_warning(func):
    def wrapper(*args):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            return func(*args)
    return wrapper


def binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate true and false positives per binary classification threshold.

    :param y_true: True labels of binary classifications
    :type y_true: np.ndarray
    :param y_score: Estimated probabilities or decision function
    :type y_score: np.ndarray
    :return:
        - fps: A count of false positives, at index i being the number of negative samples assigned a
        score >= thresholds[i]. The total number of negative samples is equal to fps[-1] (thus true negatives
        are given by fps[-1] - fps);
        - tps: An increasing count of true positives, at index i being the number of positive samples assigned a
        score >= thresholds[i]. The total number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps);
        - thresholds: Decreasing unique score values
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    pos_label = 1.0

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract the indices associated with the distinct values.
    # We also concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]

    # accumulate the true positives with decreasing threshold
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def roc(fps: np.ndarray, tps: np.ndarray, thresholds: np.ndarray, drop_intermediates: bool = False) -> np.ndarray:
    """Compute Receiver operating characteristic (ROC).

    :param fps: decreasing count of false positives
    :type fps: np.ndarray
    :param tps: increasing count of true positives
    :type tps:  np.ndarray
    :param thresholds: Decreasing thresholds on the decision function used to compute fpr and tpr. `thresholds[0]`
        represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`
    :type thresholds: np.ndarray
    :param drop_intermediates: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC
    curve. This is useful in order to create lighter  ROC curves.
    :type drop_intermediates: bool
    :return:
        - fpr: Increasing false positive rates such that element i is the false positive rate of predictions
        with score >= thresholds[i];
        - tpr: Increasing true positive rates such that element i is the true positive rate of predictions
        with score >= thresholds[i];
        - thresholds:  Decreasing thresholds on the decision function used to compute fpr and tpr. `thresholds[0]`
        represents no instances being predicted and is arbitrarily set to `max(thresholds) + 1`.

    :rtype: np.ndarray, np.ndarray, np.ndarray
    """

    if drop_intermediates is True and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Add an extra threshold to make sure that the curve starts at (0, 0)
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

    return np.array([fpr, tpr, thresholds], dtype=np.float64).round(3)


@ignore_numpy_warning
def pr(fps: np.ndarray, tps: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # last_ind = tps.searchsorted(tps[-1])
    # sl = slice(last_ind, None)

    # don't stop when full recall attained
    return np.array([np.r_[1, precision], np.r_[0, recall], np.r_[thresholds[0] + 1, thresholds]], dtype=np.float64).round(3)


def confmat(fps: np.ndarray, tps: np.ndarray) -> np.ndarray:
    # true negatives are given by
    tns = fps[-1] - fps
    # false negatives are given by
    fns = tps[-1] - tps
    # tn, fp, fn, tp
    return np.array([tns, fps, fns, tps], dtype=np.float64)


def find_length_mismatches(p: pd.DataFrame) -> List[str]:
    inconsistent_targets = []
    lbl = p.columns.get_level_values(0).unique()[1]
    for tgt, tgt_aligned in p.groupby(level=0):
        ps = tgt_aligned[(lbl, "states")].values
        rs = tgt_aligned[("ref", "states")].values
        if np.any(np.isnan(ps)) and not np.all(np.isnan(ps)):
            inconsistent_targets.append(tgt)
            logging.warning("prediction is missing some residues; {} excluded".format(tgt))
        if np.any(np.isnan(rs)):
            inconsistent_targets.append(tgt)
            logging.warning("prediction is longer than reference; {} excluded".format(tgt))

    return inconsistent_targets


def align_reference_prediction(ref: dict, pred: dict, drop_missing: bool = True) -> Tuple[pd.DataFrame, List]:
    # merge reference a prediction dicts and cast to Pandas.DataFrame
    aln_pred = pd.DataFrame({**ref, **pred})
    predname = aln_pred.columns.get_level_values(0)[-1]
    logging.debug("aligned reference and prediction; {}".format(predname))
    # check for length mismatch between reference and prediction
    wrong_len_preds = find_length_mismatches(aln_pred)
    # remove targets with length mismatch
    aln_pred = aln_pred.loc[~aln_pred.index.get_level_values(0).isin(wrong_len_preds)]
    # remove rows with nan (now it's only possible if all residues are missing)
    isnan = aln_pred.isna().all()
    isnan = isnan[isnan == 1].index.tolist()
    if isnan:
        for p in isnan:
            aln_pred[p] = aln_pred[(p[0], 'states')]

    if drop_missing is True:
        aln_pred = aln_pred.dropna(0)

    return aln_pred, wrong_len_preds


def balanced_accuracy(nd_cmat: np.ndarray) -> np.ndarray:
    c = nd_cmat.T.reshape(nd_cmat.shape[1], 2, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diagonal(c, axis1=1, axis2=2) / c.sum(axis=2)
    score = np.nanmean(per_class, axis=1)
    return score


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: Union[float, int] = 1) -> np.ndarray:
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    denom[denom == 0.] = 1  # avoid division by 0
    return (1 + beta2) * precision * recall / denom


def negative_predictive_value(tn, fn):
    denom = tn + fn
    return np.divide(tn, denom, out=np.zeros_like(tn).astype(float), where=denom != 0)


def matt_cc(tn, fp, fn, tp):
    numer = (tp*tn - fp*fn)
    denom = (np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)))
    return np.divide(numer, denom, out=np.zeros_like(numer).astype(float), where=denom != 0)


def auc(x, y):
    """ Compute Area Under the Curve (AUC) using the trapezoidal rule.

    :param x: x coordinates. These must be either monotonic increasing or monotonic decreasing
    :type x: np.ndarray
    :param y: y coordinates
    :type y: np.ndarray
    :return: area under the curve
    :rtype: float
    """

    if x.shape[0] < 2:
        logging.warning('At least 2 points are needed to compute area under curve, but x.shape = %s' % x.shape)
        area = np.nan
    else:
        direction = 1
        area = direction * np.trapz(y, x)
        if isinstance(area, np.memmap):
            # Reductions such as .sum used internally in np.trapz do not return a
            # scalar by default for numpy.memmap instances contrary to
            # regular numpy.ndarray instances.
            area = area.dtype.type(area)
    return area


def get_metrics(roc_curve, pr_curve, cmats: np.ndarray) -> dict:
    # TODO: check if it's really necessary to split.squeeze cmats
    # unpack per-threshold confusion matrices
    if cmats.shape == (4, 2):
        cmats = np.squeeze(np.split(cmats, 4, 0))
    tn, fp, fn, tp = cmats

    # remove first element (it's artificially added in pr func)
    ppv = pr_curve[0][1:]  # precision
    tpr = pr_curve[1][1:]  # sensitivity / recall

    # remove first element (they're calculated from an artificially added threshold in roc func)
    fpr = roc_curve[0][1:]  # fall-out
    tnr = 1 - fpr  # specificity / selectivity
    fnr = 1 - tpr  # miss-rate

    # compute other metrics
    bacc = balanced_accuracy(cmats)
    f1 = fbeta(ppv, tpr)
    f2 = fbeta(ppv, tpr, beta=2)
    f05 = fbeta(ppv, tpr, beta=.5)
    mcc = matt_cc(tn, fp, fn, tp)
    npv = negative_predictive_value(tn, fn)
    fom = 1 - npv  # false omission rate (for keyword is reserved)
    inf = tpr + tnr - 1  # bookmaker informedness
    mk = ppv + npv - 1  # markedness
    csi = tp / (tp + fn + fp)  # critical score index / threat score (doesn't need a func b/c denom can never be 0)

    return dict(npv=npv, ppv=ppv, tpr=tpr, tnr=tnr, fpr=fpr, fnr=fnr, fom=fom, csi=csi,
                bac=bacc, f1s=f1, f2s=f2, f05=f05, mcc=mcc, inf=inf, mk=mk)


def get_default_threshold(pred: np.ndarray) -> float:
    return pred[pred[:, 0] == 1][:, 1].min()


def calc_curves_and_metrics(ytrue, yscore):
    fps, tps, thr = binary_clf_curve(ytrue, yscore)
    roc_curve = roc(fps, tps, thr)
    pr_curve = pr(fps, tps, thr)
    cmat = confmat(fps, tps)
    metrics = get_metrics(roc_curve, pr_curve, cmat)
    return roc_curve, pr_curve, cmat, metrics


def bootstrap_reference_and_prediction(ytrue, yscore, n=100):
    for idx in (np.random.choice(len(ytrue), size=len(ytrue)) for _ in range(n)):
        ref = ytrue[idx]
        pred = yscore[idx]
        yield calc_curves_and_metrics(ref, pred)


def confidence_interval(series, interval=0.95):
    # TODO: I don't like that this function returns a pd.Series but it is necessary to have pd.DataFrame
    #  as result of an apply
    mean = series.mean()
    n = series.count()
    test_stat = stats.t.ppf((interval + 1)/2, n)
    norm_test_stat = (test_stat * series.std()) / (n ** 0.5)
    lower_bound = mean - norm_test_stat
    upper_bound = mean + norm_test_stat
    return pd.Series(dict(lo=lower_bound, hi=upper_bound))


def summary_metrics(roc_curve, pr_curve):
    ppv, tpr, _ = pr_curve
    auc_roc = auc(*roc_curve[:-1])
    auc_pr = auc(tpr, ppv)
    aps = -np.sum(np.diff(tpr[::-1]) * ppv[::-1][:-1])

    return dict(aucroc=np.round(auc_roc, 3), aucpr=np.round(auc_pr, 3), aps=np.round(aps, 3))


def dataset_curves_and_metrics(ytrue, yscore, predname):
    roc_curve, pr_curve, cmat, metrics = calc_curves_and_metrics(ytrue, yscore)
    smry_metrics = summary_metrics(roc_curve, pr_curve)

    metrics = pd.DataFrame(metrics.values(),
                           columns=roc_curve[2][1:],
                           index=pd.MultiIndex.from_product([[predname], metrics.keys()])).round(3)

    roc_df = pd.DataFrame(roc_curve[:-1].T,
                          columns=pd.MultiIndex.from_product([[predname], [smry_metrics["aucroc"]], ["fpr", "tpr"]],
                                                             names=["predictor", "auc", "metric"]),
                          index=roc_curve[-1].round(3))

    pr_df = pd.DataFrame(pr_curve[:-1].T,
                         columns=pd.MultiIndex.from_product(
                             [[predname], [smry_metrics["aucpr"]], [smry_metrics["aps"]], ["ppv", "tpr"]],
                             names=["predictor", "auc", "aps", "metric"]),
                         index=pr_curve[-1].round(3))

    cmat = pd.DataFrame(zip(*cmat),
                        columns=pd.MultiIndex.from_product([[predname], ["tn", "fp", "fn", "tp"]]),
                        index=roc_curve[-1][1:].round(3)).astype(int)


    # logging.debug("dataset metrics done")
    return roc_df, pr_df, cmat, metrics, smry_metrics


def bootstrap_curves_and_metrics(aln_refpred, predname, n):
    bootstrap_metrics = {}

    for i, data_bts in enumerate(bootstrap_reference_and_prediction(aln_refpred[('ref', 'states')].values,
                                                                    aln_refpred[(predname, 'scores')].values, n=n)):
        roc_bts, pr_bts, cmat_bts, metrics_bts = data_bts

        bts_d = {(i, m): dict(np.stack([roc_bts[2][1:], metrics_bts[m]], axis=1)) for m in metrics_bts}
        bootstrap_metrics = {**bootstrap_metrics, **bts_d}
    # save target evaluation as csv
    bootstrap_metrics = pd.DataFrame(bootstrap_metrics).round(3).T

    logging.debug("bootstrapping done")
    return bootstrap_metrics


def target_curves_and_metrics(aln_refpred, predname):
    target_metrics = {}
    for tgt, tgt_scores in aln_refpred.groupby(level=0):
        roc_tgt, pr_tgt, cmat_tgt, metrics_tgt = calc_curves_and_metrics(tgt_scores[('ref', 'states')].values,
                                                                         tgt_scores[(predname, 'scores')].values)
        # save in a data-structure easily convertible to pd.DataFrame
        tgt_d = {(tgt, m): dict(np.stack([roc_tgt[2][1:], metrics_tgt[m]], axis=1)) for m in metrics_tgt}
        # update metrics dict
        target_metrics = {**target_metrics, **tgt_d}

    target_metrics = pd.DataFrame(target_metrics).round(3).sort_index(ascending=False).fillna(method='ffill').T
    logging.debug("target metrics done")
    return target_metrics


def bvaluation(reference: str, predictions: list, outpath=".", dataset=True, target=False, bootstrap=False, run_tag="analysis"):
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    reference = Path(reference)
    refname = reference.stem
    ref_obj, accs = parse_reference(reference.resolve(strict=True))  # resolve raises an error if file doesn't exists

    roc_curves = []
    pr_curves = []
    cmats = []
    all_preds = {}
    thresholds = {}
    cm_data = {}
    dts_data = {}
    tgt_data = {}
    bts_data = {}

    for prediction in predictions:
        predname = Path(prediction).stem
        pred_obj = parse_prediction(prediction, accs, predname)  # returns dict
        aln_ref_pred, wrong_tgt = align_reference_prediction(ref_obj, pred_obj)  # remove targets w/ errors

        all_preds.update(pred_obj)  # add reference to be aligned with all preds

        roc_curve, pr_curve, cmat, dataset_metrics, smry_metrics = dataset_curves_and_metrics(
            aln_ref_pred[('ref', 'states')].values,
            aln_ref_pred[(predname, 'scores')].values,
            predname)

        if dataset is True:
            dataset_metrics.to_csv(outpath / ".".join([refname, run_tag, predname, "dataset", "metrics", "csv"]))
            roc_curves.append(roc_curve)
            pr_curves.append(pr_curve)
            cmats.append(cmat)

        if bootstrap is True:
            bootstrap_metrics = bootstrap_curves_and_metrics(aln_ref_pred, predname, 1000)
            bootstrap_metrics.to_csv(outpath / ".".join([refname, run_tag, predname, "bootstrap", "metrics", "csv"]))

        if target is True:
            target_metrics = target_curves_and_metrics(aln_ref_pred, predname)
            target_metrics.to_csv(outpath / ".".join([refname, run_tag, predname, "target", "metrics", "csv"]))

        # {<label>: <threshold>} for each threshold a file will be saved with metrics optimized for that threshold
        thresholds = {"default": get_default_threshold(aln_ref_pred[predname].to_numpy()),
                      **dataset_metrics.idxmax(1).loc[predname].to_dict()}

        # find metrics of current pred for each threshold in <thresholds>; store to be later joined with other preds
        for m in thresholds:
            if dataset is True:
                # store predictor performance in outer scope variable
                dts_data.setdefault(m, []).append(dataset_metrics[thresholds[m]].unstack().assign(**smry_metrics,
                                                                                                  thr=thresholds[m]))
                cm_data.setdefault(m, []).append(cmat.loc[thresholds[m]].unstack())
            if target is True:
                # pd.concat is a workaround to prepend a level to the existing index, creating a MultiIndex
                tgt_data.setdefault(m, []).append(pd.concat([target_metrics[thresholds[m]].unstack()],
                                                            keys=[predname]).assign(thr=thresholds[m]))
            if bootstrap is True:
                # pd.concat is a workaround to prepend a level to the existing index, creating a MultiIndex
                bts_data.setdefault(m, []).append(pd.concat(
                    [bootstrap_metrics[thresholds[m]].unstack().apply(confidence_interval).T],
                    keys=[predname]).assign(thr=thresholds[m]))

        logging.info("analysis complete; {}".format(predname))

    # merge metrics of all predictors in a pd.DataFrame; save df as csv
    for m in thresholds:
        if dataset is True:
            pd.concat(dts_data[m]).to_csv(outpath / ".".join([refname, run_tag, "all", "dataset", m, "metrics", "csv"]))
            pd.concat(cm_data[m]).to_csv(outpath / ".".join([refname, run_tag, "all", "dataset", m, "cmat", "csv"]))
        if target is True:
            pd.concat(tgt_data[m]).to_csv(outpath / ".".join([refname, run_tag, "all", "target", m, "metrics", "csv"]))
        if bootstrap is True:
            pd.concat(bts_data[m]).to_csv(outpath / ".".join([refname, run_tag, "all", "bootstrap", m, "metrics", "csv"]))

    all_preds_aligned, excluded = align_reference_prediction(ref_obj, all_preds)
    all_preds_aligned.to_csv(outpath / ".".join([refname, run_tag, "all", "dataset", "_", "predictions", "csv"]))

    if excluded:
        logging.warning("excuded targets: {}".format(", ".join(excluded)))

    if dataset is True:
        pd.concat(roc_curves, axis=1).sort_index(ascending=False)\
            .to_csv(outpath / ".".join([refname, run_tag, "all", "dataset", "_", "roc", "csv"]))
        pd.concat(pr_curves, axis=1).sort_index(ascending=False)\
            .to_csv(outpath / ".".join([refname, run_tag, "all", "dataset", "_", "pr", "csv"]))
        pd.concat(cmats, axis=1).sort_index(ascending=False)\
            .to_csv(outpath / ".".join([refname, run_tag, "all", "dataset", "_", "cmat", "csv"]))


if __name__ == "__main__":
    set_logger(0)
    allpreds = ['D001_PyHCA.out', 'D002_Predisorder.out', 'D003_IUPred2A-long.out', 'D004_IUPred2A-short.out',
                'D005_IUPred-long.out', 'D006_IUPred-short.out', 'D007_FoldUnfold.out', 'D008_IsUnstruct.out',
                'D009_GlobPlot.out', 'D010_DisPredict-2.out', 'D011_DISOPRED-3.1.out', 'D013_fIDPln.out',
                'D014_fIDPnn.out', 'D015_VSL2B.out', 'D016_DisEMBL-HL.out', 'D017_DisEMBL-465.out',
                'D018_ESpritz-D.out', 'D019_ESpritz-N.out', 'D020_ESpritz-X.out', 'D021_MobiDB-lite.out',
                'D022_S2D-1.out', 'D023_S2D-2.out', 'D024_DisoMine.out', 'D025_RawMSA.out', 'D026_AUCpreD.out',
                'D027_AUCpreD-np.out', 'D028_SPOT-Disorder1.out', 'D029_SPOT-Disorder2.out',
                'D030_SPOT-Disorder-Single.out', 'D031_JRONN.out', 'D032_DFLpred.out', 'D033_DynaMine.out']

    allpreds = [Path("data/predictions/" + p) for p in allpreds]

    bvaluation("tests/ref.test.txt", ["tests/p1.test.txt", "tests/p2.test.txt"])
    # bvaluation("data/new-disprot-all_simple.txt", allpreds, "./results", dataset=True, target=True, bootstrap=True)
    # bvaluation("data/new-disprot-all_simple.txt", ["data/predictions/D009_GlobPlot.out"])

