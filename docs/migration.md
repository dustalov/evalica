## Crowd-Kit

Users of the [Crowd-Kit](https://github.com/Toloka/crowd-kit) library can easily switch to Evalica by replacing their `label` item references with the corresponding `Winner` values, enjoying the faster and cleaner code.

```pycon
>>> import pandas as pd
>>> from crowdkit.aggregation import BradleyTerry
>>> df = pd.DataFrame(
...     [
...         ['item1', 'item2', 'item1'],
...         ['item3', 'item2', 'item2']
...     ],
...     columns=['left', 'right', 'label']
... )
>>> agg_bt = BradleyTerry(n_iter=100).fit_predict(df)
```

Evalica is not bound to the specific column names, reducing the potentially expensive operation of building a data frame, while remaining fully compatible with NumPy and pandas.

```pycon
>>> import pandas as pd
>>> from evalica import bradley_terry, Winner
>>> df = pd.DataFrame(
...     [
...         ['item1', 'item2', Winner.X],
...         ['item2', 'item3', Winner.Y]
...     ],
...     columns=['left', 'right', 'label']
... )
>>> scores = bradley_terry(df['left'], df['right'], df['label'], limit=100)
```

Or simply:

```pycon
>>> from evalica import bradley_terry, Winner
>>> scores = bradley_terry(
...     ['item1', 'item2'],
...     ['item2', 'item3'],
...     [Winner.X, Winner.Y],
...     limit=100,
... )
```

## NLTK

Users of the [NLTK](https://www.nltk.org/) library computing Krippendorff's alpha via `AnnotationTask` can switch to Evalica for a cleaner and more efficient interface.

```pycon
>>> from nltk.metrics import binary_distance
>>> from nltk.metrics.agreement import AnnotationTask
>>> data = [
...     ('coder1', 'item1', 1),
...     ('coder1', 'item2', 2),
...     ('coder2', 'item1', 1),
...     ('coder2', 'item2', 3),
...     ('coder3', 'item1', 2),
...     ('coder3', 'item2', 2),
... ]
>>> task = AnnotationTask(data, distance=binary_distance)
>>> task.alpha()
```

Evalica accepts a pandas DataFrame with observers as rows and units as columns, avoiding the need to construct `(coder, item, label)` triples manually. The built-in distance metrics are specified by name.

```pycon
>>> import pandas as pd
>>> from evalica import alpha
>>> df = pd.DataFrame(
...     [[1, 2], [1, 3], [2, 2]],
... )
>>> result = alpha(df, distance="nominal")
>>> result.alpha
```

NLTK's `binary_distance` corresponds to Evalica's `"nominal"` and `interval_distance` corresponds to `"interval"`.
