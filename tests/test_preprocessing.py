import numpy as np
import pytest

from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


@pytest.fixture
def X():
    return np.random.default_rng(42).normal(5.0, 3.0, size=(50, 4))


# StandardScaler

def test_standard_scaler_zero_mean_unit_std(X):
    Z = StandardScaler().fit_transform(X)
    assert Z.shape == X.shape
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-12)


def test_standard_scaler_constant_column_gives_zero_not_nan():
    Z = StandardScaler().fit_transform([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    assert np.isfinite(Z).all()
    np.testing.assert_array_equal(Z[:, 1], 0.0)


def test_standard_scaler_ignores_nan_when_fitting_and_propagates_it():
    X = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, 30.0]])
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.mean_, [2.0, 20.0])
    Z = sc.transform(X)
    assert np.isnan(Z[1, 0]) and np.isfinite(Z[:, 1]).all()


def test_standard_scaler_errors(X):
    with pytest.raises(ValueError, match="reshape"):
        StandardScaler().fit(X[:, 0])
    with pytest.raises(ValueError, match="features"):
        StandardScaler().fit(X).transform(X[:, :2])
    with pytest.raises(RuntimeError, match="not fitted"):
        StandardScaler().transform(X)


def test_standard_scaler_non_contiguous_input(X):
    view = X[::2]
    np.testing.assert_allclose(StandardScaler().fit_transform(view),
                               StandardScaler().fit_transform(np.ascontiguousarray(view)))


def test_standard_scaler_partial_fit_matches_batch_with_nan_and_large_offsets():
    X = np.array([
        [1.0e12 + 1, np.nan, 4.0],
        [1.0e12 + 2, 2.0, 4.0],
        [1.0e12 + 3, 4.0, 4.0],
        [1.0e12 + 4, 8.0, 4.0],
    ])
    scaler = StandardScaler().partial_fit(X[:2]).partial_fit(X[2:])
    batch = StandardScaler().fit(X)

    np.testing.assert_allclose(scaler.mean_, batch.mean_, rtol=0, atol=1e-4)
    np.testing.assert_allclose(scaler.scale_, batch.scale_, rtol=1e-5, atol=1e-8)
    np.testing.assert_array_equal(scaler.n_samples_seen_, [4, 3, 4])
    assert not np.isnan(scaler.mean_[1])
    np.testing.assert_allclose(scaler.transform(X), batch.transform(X), equal_nan=True)


def test_standard_scaler_partial_fit_all_nan_then_values_and_feature_validation():
    scaler = StandardScaler().partial_fit([[np.nan, 1.0], [np.nan, 3.0]])
    assert np.isnan(scaler.mean_[0])
    scaler.partial_fit([[5.0, 5.0]])
    np.testing.assert_allclose(scaler.mean_, [5.0, 3.0])
    with pytest.raises(ValueError, match="features"):
        scaler.partial_fit([[1.0]])
    np.testing.assert_allclose(scaler.mean_, [5.0, 3.0])


# MinMaxScaler

def test_minmax_default_and_custom_range(X):
    Z = MinMaxScaler().fit_transform(X)
    np.testing.assert_allclose(Z.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.max(axis=0), 1.0, atol=1e-12)
    Z = MinMaxScaler((-1, 1)).fit_transform(X)
    np.testing.assert_allclose(Z.min(axis=0), -1.0, atol=1e-12)
    np.testing.assert_allclose(Z.max(axis=0), 1.0, atol=1e-12)


def test_minmax_constant_column_maps_to_range_min():
    Z = MinMaxScaler((2, 5)).fit_transform([[4.0, 1.0], [4.0, 2.0], [4.0, 3.0]])
    np.testing.assert_array_equal(Z[:, 0], 2.0)


def test_minmax_invalid_range_rejected():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 1))


def test_minmax_unseen_values_extrapolate():
    sc = MinMaxScaler().fit([[0.0], [10.0]])
    np.testing.assert_allclose(sc.transform([[20.0], [-5.0]]), [[2.0], [-0.5]])


def test_minmax_partial_fit_tracks_chunk_extrema_and_nan():
    scaler = MinMaxScaler((-1, 1))
    scaler.partial_fit([[np.nan, 4.0], [np.nan, 4.0]])
    assert np.isnan(scaler.data_min_[0])
    scaler.partial_fit([[-2.0, 8.0], [6.0, np.nan]])

    np.testing.assert_allclose(scaler.data_min_, [-2.0, 4.0])
    np.testing.assert_allclose(scaler.data_range_, [8.0, 4.0])
    np.testing.assert_allclose(scaler.transform([[-2.0, 4.0], [6.0, 8.0]]), [[-1, -1], [1, 1]])
    with pytest.raises(ValueError, match="features"):
        scaler.partial_fit([[1.0]])


# SimpleImputer

def test_imputer_strategies():
    X = np.array([[1.0, 100.0], [np.nan, 1.0], [3.0, np.nan], [5.0, 2.0]])
    np.testing.assert_allclose(SimpleImputer("mean").fit(X).statistics_, [3.0, 103 / 3])
    np.testing.assert_allclose(SimpleImputer("median").fit(X).statistics_, [3.0, 2.0])
    np.testing.assert_allclose(SimpleImputer("constant", -1).fit(X).statistics_, [-1, -1])
    assert not np.isnan(SimpleImputer().fit_transform(X)).any()


def test_imputer_all_nan_column_uses_fill_value():
    imp = SimpleImputer("mean", fill_value=9.0).fit([[np.nan, 1.0], [np.nan, 2.0]])
    np.testing.assert_allclose(imp.statistics_, [9.0, 1.5])


def test_imputer_does_not_modify_input_and_rejects_bad_strategy():
    X = np.array([[np.nan, 1.0], [2.0, 3.0]])
    SimpleImputer().fit_transform(X)
    assert np.isnan(X[0, 0])
    with pytest.raises(ValueError):
        SimpleImputer(strategy="mode")


def test_imputer_fill_value_alone_implies_constant():
    X = np.array([[1.0, np.nan], [np.nan, 4.0]])
    imp = SimpleImputer(fill_value=0).fit(X)
    assert imp.strategy == "constant"
    np.testing.assert_array_equal(imp.fit_transform(X), [[1.0, 0.0], [0.0, 4.0]])
    assert SimpleImputer().strategy == "mean"


def test_imputer_string_fill_value_for_categorical_column():
    X = np.array([["Perth"], ["nan"], [""], ["Adelaide"]])
    out = SimpleImputer(fill_value="Unknown").fit_transform(X)
    assert out[:, 0].tolist() == ["Perth", "Unknown", "Unknown", "Adelaide"]
    assert OneHotEncoder().fit_transform(out).shape == (4, 3)
    with pytest.raises(ValueError):
        SimpleImputer("mean", fill_value="Unknown")


def test_imputer_mean_partial_fit_updates_from_new_observations():
    imputer = SimpleImputer("mean", fill_value=-1)
    imputer.partial_fit([[np.nan, 2.0], [np.nan, 4.0]])
    np.testing.assert_array_equal(imputer.statistics_, [-1.0, 3.0])
    imputer.partial_fit([[6.0, np.nan], [10.0, 8.0]])
    np.testing.assert_allclose(imputer.statistics_, [8.0, 14 / 3])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[8.0, 14 / 3]])


def test_imputer_median_partial_fit_is_exact_and_fit_resets_state():
    imputer = SimpleImputer("median", fill_value=9)
    imputer.partial_fit([[1.0, np.nan], [100.0, 4.0]])
    imputer.partial_fit([[2.0, 8.0], [3.0, 12.0]])
    np.testing.assert_allclose(imputer.statistics_, [2.5, 8.0])
    imputer.fit([[20.0], [30.0]])
    np.testing.assert_array_equal(imputer.statistics_, [25.0])


def test_imputer_partial_fit_rejects_feature_count_change_without_mutation():
    imputer = SimpleImputer().partial_fit([[1.0, np.nan], [3.0, 4.0]])
    before = imputer.statistics_.copy()
    with pytest.raises(ValueError, match="features"):
        imputer.partial_fit([[5.0]])
    np.testing.assert_array_equal(imputer.statistics_, before)


# OneHotEncoder

def test_onehot_string_categories_sorted_and_encoded():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]])
    enc = OneHotEncoder().fit(X)
    assert [c.tolist() for c in enc.categories_] == [["blue", "red"], ["M", "S"]]
    np.testing.assert_array_equal(enc.transform(X), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])


def test_onehot_numeric_categories_one_hot_per_column():
    Z = OneHotEncoder().fit_transform(np.array([[1, 10], [2, 10], [3, 20], [1, 20]]))
    assert Z.shape == (4, 5)
    np.testing.assert_array_equal(Z[:, :3].sum(axis=1), 1)
    np.testing.assert_array_equal(Z[:, 3:].sum(axis=1), 1)


def test_onehot_unknown_category_error_or_ignore():
    train, test = np.array([["a"], ["b"]]), np.array([["a"], ["zzz"]])
    with pytest.raises(ValueError, match="unknown"):
        OneHotEncoder().fit(train).transform(test)
    np.testing.assert_array_equal(OneHotEncoder("ignore").fit(train).transform(test), [[1, 0], [0, 0]])


def test_onehot_nan_and_blank_encode_to_zeros_and_1d_rejected():
    X = np.array([[1.0], [np.nan], [2.0]])
    np.testing.assert_array_equal(OneHotEncoder().fit_transform(X), [[1, 0], [0, 0], [0, 1]])
    np.testing.assert_array_equal(OneHotEncoder().fit_transform(np.array([["a"], [""]])), [[1], [0]])
    with pytest.raises(ValueError):
        OneHotEncoder().fit(np.array(["a", "b"]))


def test_onehot_partial_fit_appends_categories_without_shifting_columns():
    encoder = OneHotEncoder().partial_fit([["b", "x"], ["a", "y"]])
    first = encoder.transform([["a", "x"]])
    assert encoder.categories_[0].tolist() == ["a", "b"]

    encoder.partial_fit([["c", "w"], ["b", "z"]])
    assert encoder.categories_[0].tolist() == ["a", "b", "c"]
    assert encoder.categories_[1].tolist() == ["x", "y", "w", "z"]
    updated_first = encoder.transform([["a", "x"]])
    np.testing.assert_array_equal(updated_first[:, :first.shape[1]], first)
    np.testing.assert_array_equal(encoder.transform([["c", "z"]]), [[0, 0, 0, 0, 1, 0, 1]])


def test_onehot_partial_fit_feature_validation_and_fit_reset():
    encoder = OneHotEncoder().partial_fit([["a"]])
    with pytest.raises(ValueError, match="features"):
        encoder.partial_fit([["b", "c"]])
    np.testing.assert_array_equal(encoder.transform([["a"]]), [[1]])
    encoder.fit([["z"]])
    assert encoder.categories_[0].tolist() == ["z"]
