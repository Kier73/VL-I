"""Collect inference contracts from the fitted-estimator API of the pinned sklearn release."""
import inspect
import json
from pathlib import Path
import re
import sys

import sklearn
from sklearn.utils import all_estimators


def main():
    """Map each E∈all_estimators() to documented fitted attributes and inference methods."""
    if sklearn.__version__!="1.6.1":
        raise ValueError("Use scikit-learn 1.6.1 to reproduce this inventory")
    root = Path(sklearn.__file__).resolve().parent.parent
    records = []
    for name,cls in all_estimators():
        path = Path(inspect.getfile(cls)).resolve().relative_to(root).as_posix()
        docs = inspect.getdoc(cls) or ""
        methods = [m for m in ("predict","predict_proba","decision_function","transform",
                              "score_samples","kneighbors","inverse_transform") if hasattr(cls,m)]
        records.append(dict(catalogue="sklearn",entry=name,family=cls.__module__.split(".")[1],
                            source=f"https://github.com/scikit-learn/scikit-learn/blob/f159b78dc59f250cdde8fe391a21f0bc871960ad/{path}#L{inspect.getsourcelines(cls)[1]}",
                            documented_state=sorted(set(re.findall(r"(?m)^([A-Za-z]\w*_)\s*:",docs))),
                            inference_methods=methods,development="bind_estimator_state" if methods else "define_transductive_observation"))
    Path(sys.argv[1]).write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
