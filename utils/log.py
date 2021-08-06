
from typing import List

import seaborn as sn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                        class_name: List[str]) -> plt.figure:
    fig = plt.figure()
    confusion_matrix = confusion_matrix.astype(np.int64).tolist()
    df_cm = pd.DataFrame(confusion_matrix,
                        index=class_name,
                        columns=class_name)
    fig, ax = plt.subplots(figsize=(15,15))
    fig.set_tight_layout(True)
    sn.set(font_scale=1)
    svm = sn.heatmap(df_cm,
                    annot=True,
                    fmt='d',
                    annot_kws={'size': 14},
                    ax=ax,
                    cbar=False,
                    square=True)
    figure = svm.get_figure()
    return figure

