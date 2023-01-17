from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm

class Normalizer:
    def __init__(self):
        self.scaler_dict = {}
        self.scaler_dict_static = {}
        self.scaler_dict_past = {}
                 
    def normalize(self, X_static, X_time, y_past=None, with_var=True, fit=False):
        for index in tqdm(range(X_time.shape[-1])):
            if fit:
                self.scaler_dict[index] = RobustScaler(unit_variance=with_var).fit(X_time[:, :,
                                                                    index].reshape(-1, 1))
            X_time[:, :, index] = (
                self.scaler_dict[index]
                .transform(X_time[:, :, index].reshape(-1, 1))
                .reshape(-1, X_time.shape[-2])
            )
        for index in tqdm(range(X_static.shape[-1])):
            if fit:
                self.scaler_dict_static[index] = RobustScaler(unit_variance=with_var).fit(
                    X_static[:, index].reshape(-1, 1)
                )
            X_static[:, index] = (
                self.scaler_dict_static[index]
                .transform(X_static[:, index].reshape(-1, 1))
                .reshape(1, -1)
            )
        index = 0
        if y_past is not None:
            if fit:
                self.scaler_dict_past[index] = RobustScaler(unit_variance=with_var).fit(y_past.reshape(-1, 1))
            y_past[:, :] = (
                self.scaler_dict_past[index]
                .transform(y_past.reshape(-1, 1))
                .reshape(-1, y_past.shape[-1])
            )
            return X_static, X_time, y_past
        return X_static, X_time