import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


class SparseOneHotEncoder:
    '''
    A robust (works with unseen values), memory efficient and time efficient alternative to pandas.get_dummies().
    Implements sklearn's fit-transform API.
    '''
    def __init__(self):
        return

    def listify_df(self, df):
        df = df.values.tolist()
        return df

    def stringify_columns(self, df, columns, sep):
        '''
        adjust data entries to fit in format <var_name__var_value> to avoid colisions with other column entries
        :param df: data
        :param columns: categorical columns to stringify
        :param sep: separator between column name and value (default is double underscore '__')
        :return: stringyfied df
        '''
        df = df.copy()

        for column in columns:
            # standardize to float str format (to avoid mismatches like '1' != '1.0')
            try: df.loc[:, column] = df.loc[:, column].astype(float)
            except: pass

            df.loc[:, column] = column + sep + df[column].astype(str)

        return df[columns]

    def undummerize_matrix(self, matrix):
        '''
        colapse dumerized form back to undumerized form (used in inverse_transform)
        :param matrix:
        :return:
        '''
        # values_dict = {col:[] for col in self.columns}
        # values = np.array(map(lambda x: x.split(sep).__getitem__(0), matrix.flatten()))
        # keys = np.array(map(lambda x: x.split(sep).__getitem__(1), matrix.flatten()))

        # for key,value in zip(keys,values):
        #     values_dict[key].append(value)

        # return values_dict
        raise NotImplementedError('inverse transform contain errors')

    def matrix_to_df(self, matrix, columns):
        '''
        creates df containing matrix values and respective column names
        :param matrix:
        :param columns:
        :return:
        '''

        matrix = np.array(matrix)
        df = pd.DataFrame(matrix, columns=columns)
        return df

    def settify_columns(self, columns):
        '''
        assert there are no duplicate values in columns list
        :param columns:
        :return:
        '''
        if columns.__class__ not in [list, tuple, set]:
            columns = [columns]
        if columns.__class__ != set:
            columns = list(set(columns))
        return columns

    def fit(self, df, columns, sep='__', dummy_na = True):
        '''
        fit OneHotSparse object with data
        :param df: data
        :param columns: categorical columns to fit
        :param sep: separator used in new column names (category_name<sep>category_value)
        :param dummy_na: whether to create a column to nan entries
        :return: fitted OneHotSparse object
        '''
        df = df.copy()
        if dummy_na:
            df = df.append(pd.Series(), ignore_index=True)
            df = df.fillna('NaN')
        columns = self.settify_columns(columns)
        string_df = self.stringify_columns(df, columns, sep)
        df_list = self.listify_df(string_df)
        cv = CountVectorizer(tokenizer=self.tokenizer, lowercase=False, binary=True)
        cv.fit(df_list)

        self.count_vectorizer = cv
        self.categorical_features = columns
        # sorted list isntead of dict
        self.categorical_dummies = [k for k in sorted(self.count_vectorizer.vocabulary_, key=self.count_vectorizer.vocabulary_.get, reverse=False)]

        self.sep = sep
        return self

    def tokenizer(self,doc):
        '''
        tokenizer used in CountVectorizer for list entries (df_list) instead of str entries
        (lambda funcion definition causes errors in object serialization)

        :param doc: doc arg for CountVectorizer
        :return:
        '''
        return doc

    def fit_transform(self, df, columns, sep='__', return_df=True):
        df = df.copy()
        self.fit(df, columns, sep)
        transformed_result = self.transform(df, return_df)
        return transformed_result

    def assert_columns(self, df, columns):
        '''
        check if all columns passed are in df. if not, create new column and populate with ##EMPTY##.
        used in transform method to assert robustness.
        (there's probably a reason not to fill with NaN, but i can't remember why)
        :param df:
        :param columns:
        :return:
        '''
        columns_not_in_df = [i for i in columns if i not in df]
        if columns_not_in_df:
            print('{} not in DataFrame. Columns will be created with ##EMPTY## label'.format(columns_not_in_df))
            for column in columns_not_in_df:
                df[column] = '##EMPTY##'
        return df

    def transform(self, df, return_df=False):
        '''
        transform data using fitted object

        :param df:
        :param return_df: whether to return df or sparse matrix
        :return:
        '''
        df = df.copy()
        df = self.assert_columns(df, self.categorical_features)
        string_df = self.stringify_columns(df, self.categorical_features, self.sep)
        df_list = self.listify_df(string_df)
        transformed_result = self.count_vectorizer.transform(df_list)
        transformed_result = transformed_result.astype(np.int8)

        if return_df:
            dummy_df = self.matrix_to_df(transformed_result.A, self.categorical_dummies)
            transformed_result = df.assign(**dummy_df.to_dict(orient = 'list'))

        return transformed_result

    def inverse_transform(self):
        '''
        colapse from dumerized form back to categorical form

        :return:
        '''
        raise NotImplementedError('inverse transform contain errors')

