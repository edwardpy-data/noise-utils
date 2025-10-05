import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NullFunc:
    """
    Safe null-handling helper.

    - Keep usage same as before: NullFunc(df, cols=None)
    - Methods: null, not_null, def_fill, mean_fill, median_fill, mode_fill,
      forward_fill, back_fill, inter_fill
    - Behavior: If a column is entirely NaN (after any prior filtering), it will be filled with 0.
    """

    def __init__(self, df, cols=None):
        self.df = df
        self._series = not isinstance(df, pd.DataFrame)
        if not self._series:
            # Keep only columns that actually exist
            if cols:
                self.cols = [c for c in cols if c in self.df.columns]
            else:
                self.cols = list(self.df.columns)

    # ---------- small utilities ----------
    def _clean_string_like(self, s: pd.Series) -> pd.Series:
        """Normalize common textual null tokens and whitespace, return a string-dtype Series."""
        # Convert to pandas string dtype (preserves NA)
        s_str = s.astype("string")
        # normalize NBSP and strip whitespace
        s_str = s_str.str.replace("\u00A0", " ", regex=False).str.strip()
        # empty strings -> NA
        s_str = s_str.mask(s_str == "", pd.NA)
        # common textual nulls -> NA
        for token in ("nan", "NaN", "None", "NONE", "NULL", "null"):
            s_str = s_str.mask(s_str == token, pd.NA)
        return s_str

    def _is_effectively_all_nan(self, s: pd.Series) -> bool:
        """True if there are no non-null values after cleaning string-like tokens."""
        if s.dropna().empty:
            return True
        # For object/string columns check string-like tokens too
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            s2 = self._clean_string_like(s)
            return s2.dropna().empty
        return False

    # ---------- user-facing helpers ----------
    def null(self):
        """Print count of nulls per column (or for a Series)."""
        print(self.df.isnull().sum())

    def not_null(self):
        print(self.df.notnull().sum())

    def def_fill(self):
        """Fill all NaNs with 0 (in-place)."""
        # Use DataFrame.fillna to avoid chained-assignment warnings
        if self._series:
            self.df.fillna(0, inplace=True)
        else:
            self.df.fillna(0, inplace=True)

    # ---------- fill methods ----------
    def mean_fill(self):
        """Fill numeric columns with mean; non-numeric with mode; if column is all-NaN -> fill 0."""
        if self._series:
            s = self.df
            if self._is_effectively_all_nan(s):
                self.df.fillna(0, inplace=True)
                return
            if pd.api.types.is_numeric_dtype(s):
                mean_val = s.mean()
                fill_val = 0 if pd.isna(mean_val) else mean_val
                self.df.fillna(fill_val, inplace=True)
            else:
                # non-numeric series: try mode after cleaning strings
                s_clean = self._clean_string_like(s)
                modes = s_clean.mode(dropna=True)
                if not modes.empty:
                    self.df.fillna(modes.iloc[0], inplace=True)
                else:
                    self.df.fillna(0, inplace=True)
            return

        # DataFrame flow
        for col in self.cols:
            if col not in self.df.columns:
                continue
            s = self.df[col]

            # If column effectively has no valid values -> fill 0
            if self._is_effectively_all_nan(s):
                # assign via .loc to avoid SettingWithCopyWarning
                self.df.loc[:, col] = s.fillna(0)
                continue

            # If numeric dtype -> mean
            if pd.api.types.is_numeric_dtype(s):
                mean_val = s.mean()
                if pd.isna(mean_val):
                    fill_val = 0
                else:
                    fill_val = mean_val
                self.df.loc[:, col] = s.fillna(fill_val)
                continue

            # Non-numeric/mixed: attempt cleaning, then mode
            s_clean = self._clean_string_like(s)

            # If s_clean looks numeric-like (many values numeric) we can coerce to numeric
            coerced = pd.to_numeric(s_clean, errors="coerce")
            non_na = s_clean.dropna()
            coerced_non_na = coerced.dropna()
            if len(non_na) > 0 and len(coerced_non_na) >= 0.6 * len(non_na):
                # treat as numeric-like
                mean_val = coerced.mean()
                fill_val = 0 if pd.isna(mean_val) else mean_val
                self.df.loc[:, col] = coerced.fillna(fill_val)
                continue

            # otherwise fill with mode if exists, else 0
            modes = s_clean.mode(dropna=True)
            if not modes.empty:
                self.df.loc[:, col] = s_clean.fillna(modes.iloc[0])
            else:
                self.df.loc[:, col] = s_clean.fillna(0)

    def median_fill(self):
        """Same as mean_fill but uses median for numeric columns."""
        if self._series:
            s = self.df
            if self._is_effectively_all_nan(s):
                self.df.fillna(0, inplace=True)
                return
            if pd.api.types.is_numeric_dtype(s):
                med = s.median()
                fill_val = 0 if pd.isna(med) else med
                self.df.fillna(fill_val, inplace=True)
            else:
                s_clean = self._clean_string_like(s)
                modes = s_clean.mode(dropna=True)
                if not modes.empty:
                    self.df.fillna(modes.iloc[0], inplace=True)
                else:
                    self.df.fillna(0, inplace=True)
            return

        for col in self.cols:
            if col not in self.df.columns:
                continue
            s = self.df[col]
            if self._is_effectively_all_nan(s):
                self.df.loc[:, col] = s.fillna(0)
                continue
            if pd.api.types.is_numeric_dtype(s):
                med = s.median()
                fill_val = 0 if pd.isna(med) else med
                self.df.loc[:, col] = s.fillna(fill_val)
                continue
            s_clean = self._clean_string_like(s)
            modes = s_clean.mode(dropna=True)
            if not modes.empty:
                self.df.loc[:, col] = s_clean.fillna(modes.iloc[0])
            else:
                self.df.loc[:, col] = s_clean.fillna(0)

    def _mode(self, _col=None):
        """Internal mode fill helper (keeps your original naming)."""
        if self._series:
            s = self.df
            if self._is_effectively_all_nan(s):
                self.df.fillna(0, inplace=True)
                return
            s_clean = self._clean_string_like(s)
            modes = s_clean.mode(dropna=True)
            if not modes.empty:
                self.df.fillna(modes.iloc[0], inplace=True)
            else:
                self.df.fillna(0, inplace=True)
            return

        if _col is not None and _col in self.df.columns:
            s = self.df[_col]
            if self._is_effectively_all_nan(s):
                self.df.loc[:, _col] = s.fillna(0)
                return
            s_clean = self._clean_string_like(s)
            modes = s_clean.mode(dropna=True)
            if not modes.empty:
                self.df.loc[:, _col] = s_clean.fillna(modes.iloc[0])
            else:
                self.df.loc[:, _col] = s_clean.fillna(0)
        elif _col is None:
            # DataFrame mode fill columnwise
            for col in self.cols:
                self._mode(col)

    def mode_fill(self):
        """Public mode-fill using _mode() for DataFrame or Series."""
        if self._series:
            self._mode()
        else:
            for col in self.cols:
                self._mode(col)

    def forward_fill(self):
        if self._series:
            self.df.fillna(method="ffill", inplace=True)
        else:
            for col in self.cols:
                if col in self.df.columns:
                    self.df.loc[:, col] = self.df[col].fillna(method="ffill")

    def back_fill(self):
        if self._series:
            self.df.fillna(method="bfill", inplace=True)
        else:
            for col in self.cols:
                if col in self.df.columns:
                    self.df.loc[:, col] = self.df[col].fillna(method="bfill")

    def inter_fill(self):
        """Interpolate numeric-like columns; non-numeric fallback to mode or 0."""
        if self._series:
            s = self.df
            if self._is_effectively_all_nan(s):
                self.df.fillna(0, inplace=True)
                return
            if pd.api.types.is_numeric_dtype(s):
                s_interp = s.interpolate(method="linear")
                self.df.fillna(s_interp, inplace=True)
            else:
                s_clean = self._clean_string_like(s)
                modes = s_clean.mode(dropna=True)
                if not modes.empty:
                    self.df.fillna(modes.iloc[0], inplace=True)
                else:
                    self.df.fillna(0, inplace=True)
            return

        for col in self.cols:
            if col not in self.df.columns:
                continue
            s = self.df[col]
            if self._is_effectively_all_nan(s):
                self.df.loc[:, col] = s.fillna(0)
                continue
            if pd.api.types.is_numeric_dtype(s):
                self.df.loc[:, col] = s.interpolate(method="linear").fillna(s.mean() if not pd.isna(s.mean()) else 0)
            else:
                s_clean = self._clean_string_like(s)
                modes = s_clean.mode(dropna=True)
                if not modes.empty:
                    self.df.loc[:, col] = s_clean.fillna(modes.iloc[0])
                else:
                    self.df.loc[:, col] = s_clean.fillna(0)

    # ---------- convenience removal of y ----------
    def drop_nas(self, _col):
        self.df.dropna(subset = _col, inplace = True)


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        X = X.copy()
        dates = pd.to_datetime(X["Date"] + " " + X["Time"])
        X['Year'] = dates.dt.year
        X["Month"] = dates.dt.month
        X["DayOfWeek"] = dates.dt.dayofweek
        X["Day"] = dates.dt.day
        X["Hour"] = dates.dt.hour

        return X.drop(columns = ["Date","Time"])


if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\PC\Desktop\Datasets\Uber\ncr_ride_bookings.csv")

    nulls = NullFunc(df)
    nulls.drop_nas("Booking Value")
    nulls.null()
    nulls.mean_fill()
    nulls.null()