# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


from typing import List, Union, Dict
from pandas import DataFrame, Series
import torch
from torch import Tensor
import pandas as pd
from ..base import TransformerMixin


class OrdinalEncoder(TransformerMixin):
    """此处与sklearn实现不同. 详细见接口文档"""

    def __init__(self, *, categories: Union[List, List[List]] = None,
                 dtype=None, device=None):
        if isinstance(categories, List):
            # List -> List[List]
            categories = categories \
                if isinstance(categories[0], List) else [categories]
        self.categories = categories
        self.dtype = dtype
        self.device = device
        #
        self.categories_ = None  # List[List[Any]]
        self._categories_i = None  # List[Dict[Any, int]]

    def fit(self, X: Union[DataFrame, Series, List[Union[Series, List]]], y=None):
        """nan依旧用nan填充. 且不放入self.categories_中(与sklearn不同)

        :param X: shape[N] or shape[N, F]
            若为List: (列优先. 与sklearn不同). shape[F, N]
        :return:
        """
        self.dtype = self.dtype or torch.float32
        self.device = self.device or 'cpu'
        categories = self.categories
        # to DataFrame
        if isinstance(X, Series):
            X = [X]
        if isinstance(X, List):
            X = pd.DataFrame({i: s for i, s in enumerate(X)})  # 列优先
        # get categories
        if categories is None:
            categories = []
            for col in X.columns:
                Xi = X[col]  # type: Series
                #
                Xi.dropna(inplace=True)
                category = pd.unique(Xi)
                category.sort()
                categories.append(list(category))
        #
        self.categories_ = categories
        self._categories_i = [{
            c: i for i, c in enumerate(category)
        } for category in categories]
        return self

    def transform(self, X: Union[DataFrame, Series, List[Union[Series, List]]]) -> Tensor:
        """

        :param X: shape[N] or shape[N, F]
            若为List: (列优先. 与sklearn不同). shape[F, N]
        :return:
        """
        dtype = self.dtype
        device = self.device
        categories_i = self._categories_i
        # to DataFrame
        if isinstance(X, Series):
            X = [X]
        if isinstance(X, List):
            X = pd.DataFrame({i: s for i, s in enumerate(X)})  # 列优先
        #
        res = []  # 列优先
        for col, ci in zip(X.columns, categories_i):  # type: str, Dict
            Xi = X[col]
            res.append(Xi.map(ci))
        return torch.as_tensor(res, dtype=dtype, device=device).T

    def fit_transform(self, X, y=None, **fit_params) -> Tensor:
        return self.fit(X).transform(X)
