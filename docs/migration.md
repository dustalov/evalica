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
>>> scores = bradley_terry(df['left'], df['right'], df['item'], limit=100)
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
