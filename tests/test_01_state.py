import numpy as np
import pytest
from exercises.ex01_state import (normalize_quantile, unnormalize_quantile,
    discretize_state, build_prompt, encode_prompt)

def test_quantile_formula_and_no_clip():
    q01, q99 = np.array([0., 10.]), np.array([2., 14.])
    x = np.array([[0., 12.], [4., 18.]])
    y = normalize_quantile(x, q01, q99)
    expected = np.array([[-1., -0.0000002499999375],
                         [2.999998000001, 2.99999900000025]])
    np.testing.assert_allclose(y, expected, atol=1e-10)
    assert y[1, 0] > 1
    np.testing.assert_allclose(unnormalize_quantile(y, q01, q99), x, atol=1e-9)

def test_quantile_constant_dimension_is_finite():
    y = normalize_quantile(np.array([2.]), np.array([2.]), np.array([2.]))
    np.testing.assert_array_equal(y, [-1.])
    assert np.isfinite(y).all()

def test_state_bin_edges_match_repository():
    state = np.array([-2., -1., 0., .9921875, 1., 2.])
    np.testing.assert_array_equal(discretize_state(state), [-1, 0, 128, 255, 255, 255])
    np.testing.assert_array_equal(discretize_state(np.array([-1., 0., 1.]), bins=4), [0, 2, 3])

def test_prompt_state_and_advantage():
    state = np.array([-1., 0., 1.])
    assert build_prompt("  pick_up\nplug  ", state) == "Task: pick up plug, State: 0 128 255;\nAction: "
    assert build_prompt("insert", state, "positive") == (
        "Task: insert, State: 0 128 255, Advantage: positive;\nAction: ")

def test_tokenizer_callback_and_valid_zero_token():
    seen = []
    def encode(text):
        seen.append(text)
        return [2, 0, 8]
    ids, valid = encode_prompt("test prompt", encode, 5)
    assert seen == ["test prompt"]
    assert ids.dtype == np.int64 and valid.dtype == np.bool_
    np.testing.assert_array_equal(ids, [2, 0, 8, 0, 0])
    np.testing.assert_array_equal(valid, [True, True, True, False, False])

def test_tokenizer_truncation_empty_and_bad_length():
    ids, valid = encode_prompt("x", lambda _: [1, 2, 3], 2)
    np.testing.assert_array_equal(ids, [1, 2])
    assert valid.all()
    ids, valid = encode_prompt("", lambda _: [], 3)
    assert not ids.any() and not valid.any()
    with pytest.raises(ValueError):
        encode_prompt("x", lambda _: [1], 0)
