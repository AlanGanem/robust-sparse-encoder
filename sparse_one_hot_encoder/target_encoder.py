import numpy as np
from functools import partial
from .sparse_one_hot_encoder import SparseOneHotEncoder


def woe_global_part(data, target):
    tp = (data[target] == 1).sum()
    # == 0 or != 1? (takes NaNs as negative)
    tn = (data[target] != 1).sum()
    return tp, tn


def woe_calculation(data, tp, tn):
    lp = (data == 1).sum()
    ln = (data != 1).sum()
    woe = np.log((lp / tp) / (ln / tn))
    return woe


class TargetEncoder:
    '''
    performs robust target encoding for categorical variables. Targets can be numerical or categorical.
    Implements sklearn's fit-transform API.
    '''

    def __init__(self, type='categorical', sep='__', suffix = None):
        '''
        :param type: type of target variable, if not "categorical" than target will be assumed to be numerical
        :param sep: separator for new column names
        :param suffix: suffix for column name, if None, the new columns will be formed based on feature name + feature
        value for single class or numerical type. for multiclass, target class value is added too.
        '''
        self.type = type
        self.sep = sep
        self.suffix = sep + suffix if not suffix is None else ''

    def _to_col(self, x):
        '''
        support method to transform single values in lists
        :param x: value
        :return: list containing value(s)
        '''
        if x.__class__ in [list, tuple, set]:
            return list(x)
        else:
            return [x]

    def fit(self, data, columns, target, summary_func=None):
        '''
        fits encoder
        :param data: data to fit
        :param columns: categorical columns
        :param target: target columns
        :param summary_func: the function to sumarize the target for each category. if set to none, mean wil be
        performed
        :return: fitted TargetEncoder object
        '''

        data = data.copy()
        columns = self._to_col(columns)
        target = self._to_col(target)
        if self.type == 'categorical':
            self.one_hot_encoder = SparseOneHotEncoder()
            data = self.one_hot_encoder.fit_transform(data, columns=target, return_df=True)
            target = self.one_hot_encoder.categorical_dummies

        self.target_summary_mapper = {col: self._get_target_summary(data, col, target, summary_func) for col in columns}
        new_columns = [list(self.target_summary_mapper[i].columns) for i in self.target_summary_mapper]
        new_columns = sum(new_columns, [])
        self.new_columns = [i for i in new_columns if i not in self.target_summary_mapper]
        self.columns = columns
        self.target = target
        return self

    def transform(self, data, inplace=False):
        '''
        transforms data on fitted encoder
        :param data: data to be transformed
        :param inplace: whether to make transformations inplace or in a new DataFrame
        :return: transformed dataframe
        '''

        if not inplace:
            data = data.copy()

        #if self.type == 'categorical':
        #    data = self.one_hot_encoder.transform(data, return_df=True)

        for col in self.columns:
            data = data.merge(self.target_summary_mapper[col], on=col, how='left', validate='m:1')


        if self.type == 'categorical':
            data = data.drop(columns=self.target)
            
        return data

    def _get_target_summary(self, data, column, target, summary_func=None):

        new_colnames_mapper = {i: column + self.sep + i + self.suffix for i in target}
        if summary_func == 'woe':
            woe_global_part(data, target)
            tp, tn = woe_global_part(data, target)
            woe_func = partial(woe_calculation, tp = tp, tn = tn)
            transformed_data = self._local_summary_function(data, column, target, func=woe_func,
                                                            new_colnames_mapper = new_colnames_mapper)
        else:
            transformed_data = self._local_summary_function(data, column, target, func=summary_func,
                                                            new_colnames_mapper = new_colnames_mapper)

        transformed_data = transformed_data.rename(columns=new_colnames_mapper)        

        return transformed_data

    def fit_transform(self, data, columns, target, summary_func=None, inplace=False):
        self.fit(data, columns, target, summary_func)
        transformed = self.transform(data, inplace)
        return transformed

    def _local_summary_function(self, data, column, target, func, new_colnames_mapper):
        '''
        support method to get target summary for one column

        :param data: data to fit summaries from
        :param column: categorical column
        :param target: target column
        :param func: summary function (a function passed in df.groupby(columns).apply(lambda x: func(x[target))
        :return: lookup table (DataFrame) mapping categories to summaries
        '''
        data = data.copy()
        column = self._to_col(column)
        target = self._to_col(target)
        if len(column) != 1:
            raise AssertionError('must pass single value or list of single value. passed {}.'.format(column))

        if func is None:
            local_summary = data.groupby(column)[target].mean().rename(columns = new_colnames_mapper).reset_index(
                level = column)
        else:
            local_summary = data.groupby(column)[target].apply(lambda x: func(x[target])).rename(
                columns = new_colnames_mapper).reset_index(level = column)

        return local_summary




