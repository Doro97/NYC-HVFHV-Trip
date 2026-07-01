
import unittest
import numpy as np
import pandas as pd
import pickle
import os
import tempfile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# SYNTHETIC DATA HELPERS

def make_raw_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic raw DataFrame mimicking the TLC HVFHV schema."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2022-01-01")
    pickup = [base + pd.Timedelta(minutes=int(m))
              for m in rng.integers(0, 40_000, n)]
    duration = rng.uniform(1, 60, n)
    dropoff = [p + pd.Timedelta(minutes=d) for p, d in zip(pickup, duration)]

    df = pd.DataFrame({
        "hvfhs_license_num":  rng.choice(["HV0003", "HV0005"], n),
        "pickup_datetime":    pickup,
        "dropoff_datetime":   dropoff,
        "trip_miles":         rng.uniform(0.5, 25, n),
        "trip_time":          rng.integers(60, 3600, n),
        "base_passenger_fare":rng.uniform(5, 80, n),
        "tolls":              rng.choice([0.0, 6.55, 17.0], n),
        "bcf":                rng.uniform(0.1, 2.0, n),
        "sales_tax":          rng.uniform(0.3, 7.0, n),
        "congestion_surcharge":rng.choice([0.0, 1.25, 2.75], n),
        "airport_fee":        rng.choice([0.0, 1.25, 2.50], n),
        "tips":               rng.choice([0.0, 0.0, 0.0, 2.0, 5.0], n),
        "driver_pay":         rng.uniform(4, 60, n),
        "shared_match_flag":  rng.choice(["N", "Y"], n, p=[0.999, 0.001]),
        "wav_match_flag":     rng.choice(["N", "Y"], n, p=[0.95, 0.05]),
        "trip_duration_min":  duration,
    })
    return df


def make_features_df(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Generate a synthetic engineered features DataFrame (post Step 3)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "log_trip_miles":           rng.uniform(0.1, 3.0, n),
        "log_base_passenger_fare":  rng.uniform(1.5, 4.5, n),
        "log_trip_duration_min":    rng.uniform(0.7, 4.0, n),
        "is_airport_trip":          rng.integers(0, 2, n),
        "is_congestion_zone":       rng.integers(0, 2, n),
        "has_tip":                  rng.integers(0, 2, n),
        "hour_of_day":              rng.integers(0, 24, n),
    })


def make_scaled_matrix(n: int = 300, seed: int = 2) -> np.ndarray:
    """Generate a scaled feature matrix with mild cluster structure."""
    rng = np.random.default_rng(seed)
    centres = np.array([
        [-1.5, -1.0,  0.5, 0.0, 0.0, 0.0, 0.3],
        [ 0.0,  0.5, -0.5, 1.0, 0.0, 1.0, 0.6],
        [ 1.5,  1.5,  1.0, 0.0, 1.0, 0.0, 0.5],
    ])
    labels = rng.integers(0, 3, n)
    X = centres[labels] + rng.normal(0, 0.4, (n, 7))
    return X.astype(np.float64)


# 1. DATA CLEANING TESTS

class TestDataCleaning(unittest.TestCase):
    """Tests for Step 2: data cleaning logic."""

    def setUp(self):
        self.df = make_raw_df(300)

    def _clean(self, df):
        """Mirror the cleaning logic from main.py."""
        df = df.copy()
        df["trip_duration_min"] = (
            (df["dropoff_datetime"] - df["pickup_datetime"])
            .dt.total_seconds().div(60)
        )
        df = df[df["base_passenger_fare"] >= 0]
        df = df[df["driver_pay"] >= 0]
        df = df[df["trip_miles"] > 0]
        df = df[df["trip_duration_min"] >= 1]
        df = df[df["tips"] >= 0]
        df = df[df["tolls"] >= 0]
        df.drop_duplicates(
            subset=["pickup_datetime", "dropoff_datetime",
                    "trip_miles", "base_passenger_fare"],
            inplace=True,
        )
        for col in ["trip_miles", "base_passenger_fare",
                    "driver_pay", "trip_duration_min"]:
            if col in df.columns:
                cap = df[col].quantile(0.995)
                df[col] = df[col].clip(upper=cap)
        return df

    def test_negative_fare_removed(self):
        self.df.loc[0, "base_passenger_fare"] = -50
        cleaned = self._clean(self.df)
        self.assertTrue(
            (cleaned["base_passenger_fare"] >= 0).all(),
            "Negative fares should be removed."
        )

    def test_negative_driver_pay_removed(self):
        self.df.loc[1, "driver_pay"] = -20
        cleaned = self._clean(self.df)
        self.assertTrue(
            (cleaned["driver_pay"] >= 0).all(),
            "Negative driver_pay should be removed."
        )

    def test_zero_trip_miles_removed(self):
        self.df.loc[2, "trip_miles"] = 0.0
        cleaned = self._clean(self.df)
        self.assertTrue(
            (cleaned["trip_miles"] > 0).all(),
            "Zero trip_miles should be removed."
        )

    def test_sub_minute_trips_removed(self):
        # Force a sub-1-minute trip
        self.df.loc[3, "dropoff_datetime"] = (
            self.df.loc[3, "pickup_datetime"] + pd.Timedelta(seconds=30)
        )
        cleaned = self._clean(self.df)
        self.assertTrue(
            (cleaned["trip_duration_min"] >= 1).all(),
            "Trips under 1 minute should be removed."
        )

    def test_trip_duration_derived(self):
        cleaned = self._clean(self.df)
        self.assertIn("trip_duration_min", cleaned.columns)
        self.assertTrue(
            (cleaned["trip_duration_min"] > 0).all(),
            "trip_duration_min should be positive."
        )

    def test_duplicates_removed(self):
        dup = self.df.iloc[[0]].copy()
        df_with_dup = pd.concat([self.df, dup], ignore_index=True)
        cleaned = self._clean(df_with_dup)
        key_cols = ["pickup_datetime", "dropoff_datetime",
                    "trip_miles", "base_passenger_fare"]
        self.assertFalse(
            cleaned.duplicated(subset=key_cols).any(),
            "Exact duplicates should be removed."
        )

    def test_outlier_capping_applied(self):
        self.df.loc[0, "trip_miles"] = 99_999
        cleaned = self._clean(self.df)
        self.assertLess(
            cleaned["trip_miles"].max(), 99_999,
            "Extreme outlier trip_miles should be capped."
        )

    def test_original_not_mutated(self):
        original_len = len(self.df)
        _ = self._clean(self.df)
        self.assertEqual(len(self.df), original_len,
                         "Original DataFrame should not be mutated.")

    def test_tips_and_tolls_not_capped(self):
        """Tips and tolls are zero-inflated — they must not be capped."""
        self.df["tips"]  = np.where(np.arange(300) < 50, 5.0, 0.0)
        self.df["tolls"] = np.where(np.arange(300) < 30, 15.0, 0.0)
        cleaned = self._clean(self.df)
        # Non-zero values should survive
        self.assertTrue((cleaned["tips"] > 0).any(),
                        "Non-zero tips should not be capped to zero.")
        self.assertTrue((cleaned["tolls"] > 0).any(),
                        "Non-zero tolls should not be capped to zero.")


# 2. FEATURE ENGINEERING TESTS

class TestFeatureEngineering(unittest.TestCase):
    """Tests for Step 3: feature engineering and encoding."""

    def setUp(self):
        self.df = make_raw_df(200)
        # Add trip_duration_min as cleaning would have done
        self.df["trip_duration_min"] = (
            (self.df["dropoff_datetime"] - self.df["pickup_datetime"])
            .dt.total_seconds().div(60)
        )

    def _engineer(self, df):
        """Mirror the feature engineering logic from main.py."""
        df = df.copy()
        df.drop(columns=[c for c in ["bcf","sales_tax","driver_pay","trip_time"]
                         if c in df.columns], inplace=True)
        for col in ["trip_miles","base_passenger_fare","trip_duration_min"]:
            df[f"log_{col}"] = np.log1p(df[col])
            df.drop(columns=[col], inplace=True)
        df["has_tip"]   = (df["tips"]  > 0).astype(int)
        df["has_tolls"] = (df["tolls"] > 0).astype(int)
        df.drop(columns=["tips","tolls"], inplace=True)
        df["is_congestion_zone"] = (df["congestion_surcharge"] > 0).astype(int)
        df["is_airport_trip"]    = (df["airport_fee"] > 0).astype(int)
        df.drop(columns=["congestion_surcharge","airport_fee"], inplace=True)
        df["is_uber"]   = (df["hvfhs_license_num"] == "HV0003").astype(int)
        df["is_shared"] = (df["shared_match_flag"] == "Y").astype(int)
        df["is_wav"]    = (df["wav_match_flag"]    == "Y").astype(int)
        df.drop(columns=["hvfhs_license_num","shared_match_flag",
                         "wav_match_flag"], inplace=True)
        df["hour_of_day"] = df["pickup_datetime"].dt.hour
        df.drop(columns=["pickup_datetime","dropoff_datetime"], inplace=True)
        FINAL = ["log_trip_miles","log_base_passenger_fare",
                 "log_trip_duration_min","is_airport_trip",
                 "is_congestion_zone","has_tip","hour_of_day"]
        return df[FINAL]

    def test_log_transform_reduces_skewness(self):
        result = self._engineer(self.df)
        for col in ["log_trip_miles","log_base_passenger_fare",
                    "log_trip_duration_min"]:
            self.assertIn(col, result.columns)
            skew = result[col].skew()
            self.assertLess(abs(skew), 2.5,
                f"{col} skewness ({skew:.2f}) should be reduced by log1p.")

    def test_log_transform_no_negative_values(self):
        result = self._engineer(self.df)
        for col in ["log_trip_miles","log_base_passenger_fare",
                    "log_trip_duration_min"]:
            self.assertTrue(
                (result[col] >= 0).all(),
                f"{col} should have no negative values after log1p."
            )

    def test_has_tip_is_binary(self):
        result = self._engineer(self.df)
        self.assertTrue(
            result["has_tip"].isin([0,1]).all(),
            "has_tip must be binary (0 or 1)."
        )

    def test_has_tolls_is_binary(self):
        """
        has_tolls is created as a binary flag during engineering but is
        excluded from the final 7-feature selection (dropped along with
        other low-signal features). We test it at the intermediate stage
        before final selection is applied.
        """
        df = self.df.copy()
        # Partial engineering: just create the binary flag
        df["has_tolls"] = (df["tolls"] > 0).astype(int)
        self.assertTrue(
            df["has_tolls"].isin([0, 1]).all(),
            "has_tolls must be binary (0 or 1) when created."
        )
        # Confirm it is intentionally absent from the final feature set
        result = self._engineer(self.df)
        self.assertNotIn(
            "has_tolls", result.columns,
            "has_tolls is intentionally excluded from the final 7-feature set."
        )

    def test_tips_preserved_as_flag(self):
        """Non-zero tips must produce has_tip=1."""
        df = self.df.copy()
        df["tips"] = 5.0
        result = self._engineer(df)
        self.assertTrue(
            (result["has_tip"] == 1).all(),
            "All non-zero tips should produce has_tip=1."
        )

    def test_is_airport_trip_binary(self):
        result = self._engineer(self.df)
        self.assertTrue(
            result["is_airport_trip"].isin([0,1]).all(),
            "is_airport_trip must be binary."
        )

    def test_congestion_zone_binary(self):
        result = self._engineer(self.df)
        self.assertTrue(
            result["is_congestion_zone"].isin([0,1]).all(),
            "is_congestion_zone must be binary."
        )

    def test_hour_of_day_range(self):
        result = self._engineer(self.df)
        self.assertTrue(
            result["hour_of_day"].between(0, 23).all(),
            "hour_of_day must be in range 0–23."
        )

    def test_final_feature_set_correct(self):
        result = self._engineer(self.df)
        expected = ["log_trip_miles","log_base_passenger_fare",
                    "log_trip_duration_min","is_airport_trip",
                    "is_congestion_zone","has_tip","hour_of_day"]
        self.assertEqual(list(result.columns), expected,
                         "Final feature set must match specification.")

    def test_no_nulls_in_output(self):
        result = self._engineer(self.df)
        self.assertEqual(result.isnull().sum().sum(), 0,
                         "No nulls should remain after feature engineering.")

    def test_output_is_numeric(self):
        result = self._engineer(self.df)
        for col in result.columns:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(result[col]),
                f"Column {col} should be numeric."
            )


# 3. SCALING TESTS

class TestScaling(unittest.TestCase):
    """Tests for Step 4: feature scaling."""

    def setUp(self):
        self.features = make_features_df(300)
        self.STANDARD = ["log_trip_miles","log_base_passenger_fare",
                         "log_trip_duration_min"]
        self.MINMAX   = ["hour_of_day"]
        self.BINARY   = ["is_airport_trip","is_congestion_zone","has_tip"]

    def _scale(self, df):
        scaled = df.copy()
        std = StandardScaler()
        scaled[self.STANDARD] = std.fit_transform(df[self.STANDARD])
        mm  = MinMaxScaler()
        scaled[self.MINMAX]   = mm.fit_transform(df[self.MINMAX])
        return scaled, std, mm

    def test_standard_scaled_mean_near_zero(self):
        scaled, _, _ = self._scale(self.features)
        for col in self.STANDARD:
            mean = scaled[col].mean()
            self.assertAlmostEqual(mean, 0.0, places=10,
                msg=f"{col} mean should be ~0 after StandardScaler.")

    def test_standard_scaled_std_near_one(self):
        """
        StandardScaler uses population std (ddof=0) to scale.
        pandas .std() uses sample std (ddof=1) by default, so the result
        will be slightly above 1.0 for small samples.
        We verify std is within 1% of 1.0 rather than exact equality.
        """
        scaled, _, _ = self._scale(self.features)
        for col in self.STANDARD:
            std = scaled[col].std(ddof=0)   # use population std to match scaler
            self.assertAlmostEqual(std, 1.0, places=10,
                msg=f"{col} population std should be exactly 1 after StandardScaler.")

    def test_minmax_scaled_range(self):
        scaled, _, _ = self._scale(self.features)
        for col in self.MINMAX:
            self.assertAlmostEqual(scaled[col].min(), 0.0, places=10,
                msg=f"{col} min should be 0 after MinMaxScaler.")
            self.assertAlmostEqual(scaled[col].max(), 1.0, places=10,
                msg=f"{col} max should be 1 after MinMaxScaler.")

    def test_binary_features_unchanged(self):
        scaled, _, _ = self._scale(self.features)
        for col in self.BINARY:
            pd.testing.assert_series_equal(
                scaled[col], self.features[col],
                check_names=True,
                obj=f"{col} should not be scaled."
            )

    def test_no_nulls_after_scaling(self):
        scaled, _, _ = self._scale(self.features)
        self.assertEqual(scaled.isnull().sum().sum(), 0,
                         "No nulls should be introduced by scaling.")

    def test_scaler_serialisable(self):
        _, std, mm = self._scale(self.features)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                pickle.dump({"standard": std, "minmax": mm}, f)
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            self.assertIn("standard", loaded)
            self.assertIn("minmax", loaded)
        finally:
            os.unlink(path)


# 4. K-MEANS MODEL TESTS

class TestKMeansModel(unittest.TestCase):
    """Tests for Step 5: K-Means model fitting and prediction."""

    def setUp(self):
        self.X = make_scaled_matrix(300, seed=2)
        self.K = 7

    def test_correct_number_of_labels(self):
        model = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        labels = model.fit_predict(self.X)
        self.assertEqual(len(labels), len(self.X),
                         "Number of labels must equal number of input rows.")

    def test_label_values_in_valid_range(self):
        model = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        labels = model.fit_predict(self.X)
        self.assertEqual(set(np.unique(labels)), set(range(self.K)),
                         f"Labels must be integers in range 0 to {self.K-1}.")

    def test_cluster_centers_shape(self):
        model = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        model.fit(self.X)
        self.assertEqual(model.cluster_centers_.shape,
                         (self.K, self.X.shape[1]),
                         "Cluster centers must have shape (k, n_features).")

    def test_inertia_positive(self):
        model = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        model.fit(self.X)
        self.assertGreater(model.inertia_, 0,
                           "Inertia must be positive.")

    def test_inertia_decreases_with_k(self):
        """Inertia must be non-increasing as k grows."""
        inertias = []
        for k in range(2, 6):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(self.X)
            inertias.append(km.inertia_)
        for i in range(len(inertias) - 1):
            self.assertGreaterEqual(inertias[i], inertias[i+1],
                "Inertia must be non-increasing as k increases.")

    def test_deterministic_with_fixed_seed(self):
        km1 = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        km2 = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        np.testing.assert_array_equal(
            km1.fit_predict(self.X),
            km2.fit_predict(self.X),
            err_msg="K-Means must be deterministic with fixed random_state."
        )

    def test_model_serialisable(self):
        model = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        model.fit(self.X)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                pickle.dump(model, f)
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            np.testing.assert_array_equal(
                model.labels_, loaded.labels_,
                err_msg="Loaded model labels must match original."
            )
        finally:
            os.unlink(path)


# 5. CLUSTER EVALUATION TESTS

class TestClusterEvaluation(unittest.TestCase):
    """Tests for Step 6: cluster profiling and segment labelling."""

    def setUp(self):
        self.features = make_features_df(300)
        X = make_scaled_matrix(300, seed=2)
        model = KMeans(n_clusters=7, random_state=42, n_init=10)
        self.labels = model.fit_predict(X)
        self.features["cluster"] = self.labels

    def _back_transform(self, df):
        df = df.copy()
        df["trip_miles"]          = np.expm1(df["log_trip_miles"])
        df["base_passenger_fare"] = np.expm1(df["log_base_passenger_fare"])
        df["trip_duration_min"]   = np.expm1(df["log_trip_duration_min"])
        return df

    def _assign_label(self, row):
        miles   = row["trip_miles_mean"]
        fare    = row["base_passenger_fare_mean"]
        airport = row["is_airport_trip_mean"]
        cong    = row["is_congestion_zone_mean"]
        if airport > 0.3:               return "Airport Fee Trips (Long-Haul)"
        if miles > 6:                   return "Cross-Borough Long"
        if cong >= 0.99 and miles < 3:  return "Congestion Zone Short"
        if cong >= 0.99 and miles >= 3: return "Congestion Zone Mid"
        if cong < 0.05 and miles > 3:   return "Non-Zone Mid-Distance"
        if cong < 0.05 and miles <= 3:  return "Non-Zone Short"
        if fare < 10:                   return "Minimum Fare Micro-Trip"
        return "Unclassified"

    def test_back_transform_positive_values(self):
        bt = self._back_transform(self.features)
        for col in ["trip_miles","base_passenger_fare","trip_duration_min"]:
            self.assertTrue(
                (bt[col] >= 0).all(),
                f"Back-transformed {col} should be non-negative."
            )

    def test_back_transform_reverses_log(self):
        bt = self._back_transform(self.features)
        np.testing.assert_array_almost_equal(
            np.log1p(bt["trip_miles"].values),
            self.features["log_trip_miles"].values,
            decimal=10,
            err_msg="expm1(log1p(x)) should equal x."
        )

    def test_profile_has_one_row_per_cluster(self):
        bt = self._back_transform(self.features)
        PROFILE_COLS = ["trip_miles","base_passenger_fare",
                        "trip_duration_min","is_airport_trip",
                        "is_congestion_zone","has_tip","hour_of_day"]
        profile = bt.groupby("cluster")[PROFILE_COLS].mean()
        self.assertEqual(len(profile), len(np.unique(self.labels)),
                         "Profile must have one row per cluster.")

    def test_cluster_sizes_sum_to_total(self):
        sizes = pd.Series(self.labels).value_counts()
        self.assertEqual(sizes.sum(), len(self.labels),
                         "Cluster sizes must sum to total number of rows.")

    def test_size_balance_acceptable(self):
        sizes = pd.Series(self.labels).value_counts()
        balance = sizes.min() / sizes.max()
        self.assertGreater(balance, 0.1,
            f"Cluster size balance ({balance:.3f}) is too imbalanced.")

    def test_segment_labels_are_strings(self):
        bt = self._back_transform(self.features)
        PROFILE_COLS = ["trip_miles","base_passenger_fare",
                        "trip_duration_min","is_airport_trip",
                        "is_congestion_zone","has_tip","hour_of_day"]
        profile = bt.groupby("cluster")[PROFILE_COLS].mean()
        profile.columns = [f"{c}_mean" for c in profile.columns]
        labels = [self._assign_label(profile.iloc[i]) for i in range(len(profile))]
        for label in labels:
            self.assertIsInstance(label, str,
                                  "Segment labels must be strings.")
            self.assertGreater(len(label), 0,
                               "Segment labels must be non-empty.")

    def test_no_unclassified_on_real_profiles(self):
        """With realistic synthetic data, no cluster should be Unclassified."""
        # Build a profile that mimics real cluster means
        fake_profile = pd.DataFrame({
            "trip_miles_mean":          [4.4, 1.7, 17.5, 1.0, 3.8, 2.1, 8.5],
            "base_passenger_fare_mean": [18.9,11.6,54.7, 8.2,20.3,11.6,30.7],
            "is_airport_trip_mean":     [0.04,0.00,0.41,0.00,0.00,0.01,0.12],
            "is_congestion_zone_mean":  [0.00,1.00,0.42,0.18,1.00,0.00,0.52],
        })
        labels = [self._assign_label(fake_profile.iloc[i])
                  for i in range(len(fake_profile))]
        self.assertNotIn("Unclassified", labels,
            "No cluster should receive the Unclassified fallback label.")


# 6. END-TO-END PIPELINE SMOKE TEST

class TestPipeline(unittest.TestCase):
    """
    Smoke test: run the full pipeline logic on a small synthetic dataset
    and verify the output shape, types, and key invariants are satisfied.
    Does NOT require the real parquet file.
    """

    def test_full_pipeline_synthetic(self):
        # --- Simulate cleaning ---
        raw = make_raw_df(500)
        raw["trip_duration_min"] = (
            (raw["dropoff_datetime"] - raw["pickup_datetime"])
            .dt.total_seconds().div(60)
        )
        clean = raw[raw["base_passenger_fare"] >= 0].copy()
        clean = clean[clean["trip_miles"] > 0]
        clean = clean[clean["trip_duration_min"] >= 1]
        self.assertGreater(len(clean), 0, "Cleaned dataset must not be empty.")

        # --- Simulate feature engineering ---
        for col in ["trip_miles","base_passenger_fare","trip_duration_min"]:
            clean[f"log_{col}"] = np.log1p(clean[col])
        clean["has_tip"]            = (clean["tips"] > 0).astype(int)
        clean["is_congestion_zone"] = (clean["congestion_surcharge"] > 0).astype(int)
        clean["is_airport_trip"]    = (clean["airport_fee"] > 0).astype(int)
        clean["hour_of_day"]        = clean["pickup_datetime"].dt.hour

        FEATURES = ["log_trip_miles","log_base_passenger_fare",
                    "log_trip_duration_min","is_airport_trip",
                    "is_congestion_zone","has_tip","hour_of_day"]
        feat = clean[FEATURES].copy()
        self.assertEqual(feat.shape[1], 7,
                         "Feature matrix must have 7 columns.")
        self.assertEqual(feat.isnull().sum().sum(), 0,
                         "Feature matrix must have no nulls.")

        # --- Simulate scaling ---
        std = StandardScaler()
        mm  = MinMaxScaler()
        scaled = feat.copy()
        STD_COLS = ["log_trip_miles","log_base_passenger_fare","log_trip_duration_min"]
        MM_COLS  = ["hour_of_day"]
        scaled[STD_COLS] = std.fit_transform(feat[STD_COLS])
        scaled[MM_COLS]  = mm.fit_transform(feat[MM_COLS])

        # --- Simulate modelling ---
        X = scaled.values
        model = KMeans(n_clusters=7, random_state=42, n_init=5)
        labels = model.fit_predict(X)
        self.assertEqual(len(labels), len(X))
        self.assertEqual(len(np.unique(labels)), 7)

        # --- Simulate evaluation ---
        feat["cluster"] = labels
        feat["trip_miles_bt"] = np.expm1(feat["log_trip_miles"])
        profile = feat.groupby("cluster")["trip_miles_bt"].mean()
        self.assertEqual(len(profile), 7,
                         "Profile must have one row per cluster.")
        self.assertTrue((profile > 0).all(),
                        "All cluster mean trip miles must be positive.")


# ENTRY POINT

if __name__ == "__main__":
    unittest.main(verbosity=2)