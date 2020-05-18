

from .sparse_onehot_encoder import RobustOneHotSparse

class TargetEncoder:
    '''
    performs robust target encoding for categorical variables. Targets can be numerical or categorical.
    Implements sklearn's fit-transform API.
    '''

    def __init__(self, type = 'categorical', sep = '__'):
        self.type = type
        self.sep = sep

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

    def fit(self, data, columns, target, summary_func = None):
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
            self.one_hot_encoder = RobustOneHotSparse()
            data = self.one_hot_encoder.fit_transform(data, columns= target)
            target = self.one_hot_encoder.categorical_dummies

        self.target_summary_mapper = {col:self._get_target_summary(data, col, target, summary_func) for col in columns}
        self.columns = columns
        self.target = target
        return self

    def transform(self, data, inplace = False):
        '''
        transforms data on fitted encoder
        :param data: data to be transformed
        :param inplace: whether to make transformations inplace or in a new DataFrame
        :return: transformed dataframe
        '''

        if not inplace:
            data = data.copy()
        if self.type == 'categorical':
            data = self.one_hot_encoder.transform(data)

        for col in self.columns:
            data = data.merge(self.target_summary_mapper[col], on = col, how = 'left', validate = 'm:1')

        if self.type == 'categorical':
            data = data.drop(columns = self.target)

        return data

    def _get_target_summary(self, data, columns, target, func = None):
        '''
        support method to get target summary for one column

        :param data: data to fit summaries from
        :param columns: categorical column
        :param target: target column
        :param func: summary function (a function passed in df.groupby(columns).apply(lambda x: func(x[target))
        :return: lookup table (DataFrame) mapping categories to summaries
        '''
        data = data.copy()
        columns = self._to_col(columns)
        target = self._to_col(target)
        if len(columns) != 1:
            raise AssertionError('must pass single value or list of single value. passed {}.'.format(columns))

        new_colnames_mapper = {i:columns[0] + self.sep + i for i in target}
        if not func:
            target_summary = data.groupby(columns)[target].mean().rename(columns = new_colnames_mapper).reset_index()
        else:
            target_summary = data.groupby(columns)[target].apply(lambda x: func(x[target])).rename(
                columns = new_colnames_mapper).reset_index()

        return target_summary