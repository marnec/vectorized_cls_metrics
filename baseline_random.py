import logging
import numpy as np
import pandas as pd
from pathlib import Path
from parsers import parse_reference

# relative imports
from logger import set_logger
from vectorized_metrics import dataset_curves_and_metrics, target_curves_and_metrics


def baseline_random(ref, n=100, basename="", target=True, outpath='.'):
    print("baseline random")
    """Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
    """
    logging.info("Calculating baseline random")
    outpath = Path(outpath)
    bsl_data = {}
    tgt_data = {}
    thresholds = {}
    ytrue = ref.values[:, 0]

    for i, yscore in enumerate(np.random.rand(n, len(ytrue)).round(3)):
        predname = "random{}".format(i)
        logging.debug("random {} start".format(i))
        *_, metrics, smry_metrics = dataset_curves_and_metrics(ytrue, yscore, predname)
        logging.debug("dataset {} done".format(1))

        if target is True:  # very slow
            aln_ref_pred = ref.assign(**{predname: yscore})  # **{} allows to dynamically assign column name
            print("nproteins", len(aln_ref_pred.index.get_level_values(0).unique()))
            aln_ref_pred.columns.set_levels(list(ref.columns.get_level_values(1)) + ["scores"], level=1, inplace=True)
            target_metrics = target_curves_and_metrics(aln_ref_pred, predname)
            # target_metrics.to_csv(outpath / ".".join([basename, "random", predname, "target", "metrics", "csv"]))

        thresholds = {"default": yscore.flat[np.abs(yscore - 0.5).argmin()],
                      **metrics.idxmax(1).loc[predname].to_dict()}

        for m in thresholds:
            # store predictor performance in outer scope variable
            bsl_data.setdefault(m, []).append(metrics[thresholds[m]].unstack().assign(**smry_metrics,
                                                                                      thr=thresholds[m]))
            if target is True:
                # pd.concat is a workaround to prepend a level to the existing index, creating a MultiIndex
                tgt_data.setdefault(m, []).append(pd.concat([target_metrics[thresholds[m]].unstack()],
                                                            keys=[predname]).assign(thr=thresholds[m]))
        logging.debug("target {} done".format(i))

    for m in thresholds:
        df = pd.concat(bsl_data[m])
        df.to_csv(outpath / ".".join([basename, "random", "all", "dataset", m, "metrics", "csv"]))
        df.describe().round(3).to_csv(outpath / ".".join([basename, "random", "avg", "dataset", m, "metrics", "csv"]))
        logging.debug("dataset csv written for threshold".format(m))
        if target is True:
            pd.concat(tgt_data[m]).to_csv(outpath / ".".join([basename, "random", "all", "target", m, "metrics", "csv"]))
            df.describe().round(3).to_csv(
                outpath / ".".join([basename, "random", "avg", "target", m, "metrics", "csv"]))
        logging.debug("target csv written for threshold".format(m))

def baseline_shuffle_dataset(ref, n=100, basename="", target=True, outpath="."):
    outpath = Path(outpath)
    bsl_data = {}
    tgt_data = {}
    thresholds = {}
    ytrue = ref.values[:, 0]

    logging.info("Calculating baseline shuffle-datasets")
    size = len(ytrue)
    for i, yscore in enumerate((np.random.choice(ytrue, replace=False, size=size) for _ in range(n))):
        logging.debug("random {} start".format(i))
        predname = "shuffledataset{}".format(i)
        *_, metrics, smry_metrics = dataset_curves_and_metrics(ytrue, yscore, predname)

        if target is True:  # very slow
            aln_ref_pred = ref.assign(**{predname: yscore})  # **{} allows to dynamically assign column name
            aln_ref_pred.columns.set_levels(list(ref.columns.get_level_values(1)) + ["scores"], level=1, inplace=True)
            target_metrics = target_curves_and_metrics(aln_ref_pred, predname)

        thresholds = {"default": 1.0,
                      **metrics.idxmax(1).loc[predname].to_dict()}

        logging.debug("target {} done".format(i))

        for m in thresholds:
            # store predictor performance in outer scope variable
            bsl_data.setdefault(m, []).append(metrics[thresholds[m]].unstack().assign(**smry_metrics,
                                                                                      thr=thresholds[m]))
            if target is True:
                # pd.concat is a workaround to prepend a level to the existing index, creating a MultiIndex
                tgt_data.setdefault(m, []).append(pd.concat([target_metrics[thresholds[m]].unstack()],
                                                            keys=[predname]).assign(thr=thresholds[m]))

    for m in thresholds:
        df = pd.concat(bsl_data[m])
        df.to_csv(outpath / ".".join([basename, "shuffledataset", "all", "dataset", m, "metrics", "csv"]))
        df.describe().round(3).to_csv(outpath / ".".join([basename, "shuffledataset", "avg", "dataset", m, "metrics", "csv"]))
        logging.debug("dataset csv written for threshold".format(m))
        if target is True:
            pd.concat(tgt_data[m]).to_csv(outpath / ".".join([basename, "shuffledataset", "all", "target", m, "metrics", "csv"]))
            df.describe().round(3).to_csv(
                outpath / ".".join([basename, "shuffledataset", "avg", "target", m, "metrics", "csv"]))

        logging.debug("target csv written for threshold".format(m))


def baseline_shuffle_targets(ref, n=100, basename="", target=True, outpath="."):
    outpath = Path(outpath)
    ytrue = ref[("ref", "states")].values
    bsl_data = {}
    tgt_data = {}
    thresholds = {}


    logging.info("Calculating baseline shuffle-target")

    for i in range(n):
        predname = "shuffletargets{}".format(i)

        yscore = []
        for _, group in ref.groupby(level=0):
            yscore.append(np.random.choice(group[("ref", "states")].values, replace=False, size=len(group)))
        yscore = np.concatenate(yscore)

        if target is True:  # very slow
            aln_ref_pred = ref.assign(**{predname: yscore})  # **{} allows to dynamically assign column name
            aln_ref_pred.columns.set_levels(list(ref.columns.get_level_values(1)) + ["scores"], level=1, inplace=True)
            target_metrics = target_curves_and_metrics(aln_ref_pred, predname)

        *_, metrics, smry_metrics = dataset_curves_and_metrics(ytrue, yscore, predname)

        thresholds = {"default": 1.0,
                      **metrics.idxmax(1).loc[predname].to_dict()}

        for m in thresholds:
            # store predictor performance in outer scope variable
            bsl_data.setdefault(m, []).append(metrics[thresholds[m]].unstack().assign(**smry_metrics,
                                                                                      thr=thresholds[m]))

            if target is True:
                # pd.concat is a workaround to prepend a level to the existing index, creating a MultiIndex
                tgt_data.setdefault(m, []).append(pd.concat([target_metrics[thresholds[m]].unstack()],
                                                            keys=[predname]).assign(thr=thresholds[m]))

    for m in thresholds:
        df = pd.concat(bsl_data[m])
        df.to_csv(outpath / ".".join([basename, "shuffletargets", "all", "dataset", m, "metrics", "csv"]))
        df.describe().round(3).to_csv(outpath / ".".join([basename, "shuffletargets", "avg", "dataset", m, "metrics", "csv"]))

        if target is True:
            pd.concat(tgt_data[m]).to_csv(outpath / ".".join([basename, "shuffletargets", "all", "target", m, "metrics", "csv"]))
            df.describe().round(3).to_csv(
                outpath / ".".join([basename, "shuffletargets", "avg", "target", m, "metrics", "csv"]))


def baseline_fixed_positive_fraction(ref, f, n=100, basename="", target=True, outpath="."):
    outpath = Path(outpath)
    bsl_data = {}
    tgt_data = {}
    thresholds = {}
    ytrue = ref.values[:, 0]

    logging.info("Calculating baseline fixed-positive-fraction")

    size = len(ytrue)
    for i, yscore in enumerate(np.greater(np.random.rand(size), 1-f).astype(int) for _ in range(n)):
        predname = "fixedposfrc{}".format(i)
        *_, metrics, smry_metrics = dataset_curves_and_metrics(ytrue, yscore, predname)

        if target is True:  # very slow
            aln_ref_pred = ref.assign(**{predname: yscore})  # **{} allows to dynamically assign column name
            aln_ref_pred.columns.set_levels(list(ref.columns.get_level_values(1)) + ["scores"], level=1, inplace=True)
            target_metrics = target_curves_and_metrics(aln_ref_pred, predname)

        thresholds = {"default": 1.0,
                      **metrics.idxmax(1).loc[predname].to_dict()}

        for m in thresholds:
            # store predictor performance in outer scope variable
            bsl_data.setdefault(m, []).append(metrics[thresholds[m]].unstack().assign(**smry_metrics,
                                                                                      thr=thresholds[m]))
            if target is True:
                # pd.concat is a workaround to prepend a level to the existing index, creating a MultiIndex
                tgt_data.setdefault(m, []).append(pd.concat([target_metrics[thresholds[m]].unstack()],
                                                            keys=[predname]).assign(thr=thresholds[m]))

    for m in thresholds:
        df = pd.concat(bsl_data[m])
        df.to_csv(outpath / ".".join([basename, "fixedposfrc", "all", "dataset", m, "metrics", "csv"]))
        df.describe().round(3).to_csv(outpath / ".".join([basename, "fixedposfrc", "avg", "dataset", m, "metrics", "csv"]))

        if target is True:
            pd.concat(tgt_data[m]).to_csv(outpath / ".".join([basename, "fixedposfrc", "all", "target", m, "metrics", "csv"]))
            df.describe().round(3).to_csv(
                outpath / ".".join([basename, "fixedposfrc", "avg", "target", m, "metrics", "csv"]))


def get_reference(reference):
    logging.info("getting reference: {}".format(reference))
    reference = Path(reference)
    refname = reference.stem
    ref_obj, accs = parse_reference(reference.resolve(strict=True))  # resolve raises an error if file doesn't exists

    return pd.DataFrame(ref_obj), refname


if __name__ == "__main__":
    set_logger("INFO")
    ref, refname = get_reference("tests/ref.test.txt")
    # baseline_random(ref, basename=refname)
    # baseline_shuffle_dataset(ref, basename=refname)
    # baseline_shuffle_targets(ref, basename=refname)
    baseline_fixed_positive_fraction(ref, 0.3, basename=refname)