import numpy as np

from sklearn.linear_model import HuberRegressor
from sklearn.utils import resample


class BootstrappedRegressor:
    def __init__(
        self,
        model_cls=HuberRegressor,
        n_bootstrap=100,
        random_state=None,
        **model_kwargs,
    ):
        self.model_cls = model_cls
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self.boot_slopes = []
        self.slope_mean = None
        self.slope_std = None
        self.boot_intercepts = []

    def fit(self, x, y):
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_bootstrap):
            x_boot, y_boot = resample(x, y, random_state=rng)
            try:
                model = self.model_cls(**self.model_kwargs).fit(x_boot, y_boot)
                self.boot_slopes.append(model.coef_[0])
                self.boot_intercepts.append(model.intercept_)
            except Exception as e:
                print(f"Error during bootstrapping: {e}")
                continue

        self.boot_slopes = np.array(self.boot_slopes)
        self.boot_intercepts = np.array(self.boot_intercepts)

        self.slope_mean = np.mean(self.boot_slopes)
        self.slope_std = np.std(self.boot_slopes, ddof=1)
        self.slope_med = np.median(self.boot_slopes)

        self.intercept_mean = np.mean(self.boot_intercepts)
        self.intercept_std = np.std(self.boot_intercepts, ddof=1)
        self.intercept_med = np.median(self.boot_intercepts)

    def predict_line(self, x_range):
        return self.slope_med * x_range + self.intercept_med

    def get_slope_ci(self, ci=95):
        ps = 50 - ci / 2, 50 + ci / 2
        return np.nanpercentile(self.boot_slopes, ps)

    def get_intercept_ci(self, ci=95):
        ps = 50 - ci / 2, 50 + ci / 2
        return np.nanpercentile(self.boot_intercepts, ps)

    # def get_eps_estimate(self):
    #     if self.slope_mean > 0:
    #         eps = self.slope_mean ** (-1 / 4)
    #         eps_err = (1 / 4) * (1 / (self.slope_mean ** (5 / 4))) * self.slope_std
    #         return eps, eps_err
    #     else:
    #         return np.nan, np.nan

    def predict_band(self, x_range, ci=95, estimator=np.median):
        """Return mean prediction and CI band over bootstrapped fits."""
        if len(self.boot_slopes) == 0:
            raise ValueError("Model not fitted")

        intercepts = np.array(self.boot_intercepts)
        slopes = np.array(self.boot_slopes)
        preds = (
            np.outer(slopes, x_range) + intercepts[:, np.newaxis]
        )  # shape: (B, len(x_range))

        y_mean = estimator(preds, axis=0)
        lower = np.percentile(preds, (100 - ci) / 2, axis=0)
        upper = np.percentile(preds, 100 - (100 - ci) / 2, axis=0)

        return y_mean, lower, upper
