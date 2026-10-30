import itertools

import numpy as np
import pytest

from lbf import (
    SMOOTH,
    PaperProtocol,
    Protocol,
    coefficients,
    equal_length_lambdas,
    load_run,
    mix_replicas,
    smooth_step,
    switch,
)


def finite_difference(f, x, e=1e-6):
    return (f(x + e) - f(x - e)) / (2 * e)


def test_known_smoothsteps():
    assert coefficients(2, 2) == (10, -15, 6)
    assert coefficients(3, 3) == (35, -84, 70, -20)
    assert coefficients(0, 0) == (1,)


def test_boundary_conditions():
    for m0 in range(8):
        for m1 in range(4):
            s, ds = switch(m0, m1, np.array([0.0, 1.0]))
            assert s[0] == 0 and s[1] == 1
            if m1 >= 1:
                assert abs(ds[1]) < 1e-9
            if m0 >= 1:
                assert ds[0] == 0


def test_switch_derivative():
    x = np.linspace(0.05, 0.95, 7)
    for m0, m1 in [(0, 1), (2, 0), (4, 2)]:
        assert np.allclose(switch(m0, m1, x)[1], finite_difference(lambda y: switch(m0, m1, y)[0], x), atol=1e-6)


def test_smooth_step():
    x = np.linspace(0.02, 0.98, 9)
    assert np.allclose(smooth_step(x)[1], finite_difference(lambda y: smooth_step(y)[0], x), atol=1e-6)
    assert np.allclose(smooth_step(x)[0] + smooth_step(1 - x)[0], 1.0)
    s, ds = smooth_step(np.array([0.0, 1e-3, 1.0]))
    assert s[0] == 0 and s[1] < 1e-300 and s[2] == 1 and ds[0] == 0


@pytest.mark.parametrize("text", ["C(1,0) D(11,1)[0.25] E(2,0)", "C(0,2) D(inf)[0.25] E(5,1)"])
def test_protocol_text(text):
    protocol = Protocol.parse(text)
    assert str(protocol) == text
    assert Protocol.parse(str(protocol)) == protocol
    assert Protocol.parse(text.replace(" ", "")) == protocol


@pytest.mark.parametrize("text", ["C(1,0) D(11,1)", "C(1,0) D(11,1) E(2,0) X", "C(1,0) D(11) E(2,x)"])
def test_protocol_errors(text):
    with pytest.raises(ValueError):
        Protocol.parse(text)


@pytest.mark.parametrize("protocol", [Protocol.parse("C(1,0) D(11,1)[0.25] E(2,0)"),
                                      Protocol.parse("C(1,1) D(inf)[0.25] E(2,1)"), PaperProtocol()])
def test_protocol_h(protocol):
    lam = np.linspace(0, 1, 201)
    h, dh = protocol.h(lam)
    assert h.shape == dh.shape == (3, len(lam))
    assert np.allclose(h[:, 0], 0) and np.allclose(h[:, -1], 1)
    smooth = (lam > 0.01) & (lam < 0.99) & np.all(np.abs(lam[:, None] - [0.25, 0.4, 0.6, 0.7, 0.8]) > 0.01, axis=1)
    fd = finite_difference(lambda x: protocol.h(x)[0], lam)
    assert np.allclose(dh[:, smooth], fd[:, smooth], rtol=1e-5, atol=1e-7)


def test_window():
    h, dh = Protocol.parse("C(1,0) D(11,1)[0.25] E(2,0)").h(np.linspace(0, 1, 101))
    lam = np.linspace(0, 1, 101)
    assert np.all(h[1, lam <= 0.25] == 0) and np.all(dh[1, lam <= 0.25] == 0)
    assert np.all(h[1, lam > 0.25] > 0)


def test_equal_length_lambdas():
    lam = np.linspace(0, 1, 1001)
    assert np.allclose(equal_length_lambdas(lam, np.ones_like(lam), 11), np.linspace(0, 1, 11))
    # sd = 2 lambda: the length up to lambda is lambda^2
    assert np.allclose(equal_length_lambdas(lam, 2 * lam, 6), np.sqrt(np.linspace(0, 1, 6)), atol=1e-4)
    # no length at all over the first half: every state but the first lies in the second half
    places = equal_length_lambdas(lam, (lam > 0.5).astype(float), 5)
    assert places[0] == 0 and np.all(places[1:] > 0.5) and places[-1] == 1


def test_mix_replicas_samples_the_permutation_distribution():
    # Exchanges leave p(sigma) ~ exp(-sum_i u[i, sigma(i)]) invariant; with the configurations held fixed, the
    # chain of permutations must visit each one with that probability.
    rng = np.random.default_rng(7)
    num = 4
    u = rng.normal(scale=1.5, size=(num, num))
    permutations = list(itertools.permutations(range(num)))
    weights = np.array([np.exp(-sum(u[i, s[i]] for i in range(num))) for s in permutations])
    exact = weights / weights.sum()
    state = np.arange(num)
    counts = dict.fromkeys(permutations, 0)
    accepted, attempted = np.zeros((num, num), int), np.zeros((num, num), int)
    samples = 40000
    for _ in range(samples):
        mix_replicas(u, state, 8, rng, accepted, attempted)
        counts[tuple(state)] += 1
    observed = np.array([counts[s] for s in permutations]) / samples
    assert np.allclose(observed, exact, atol=0.01)
    assert np.all(accepted <= attempted) and np.array_equal(attempted, attempted.T)
    assert sorted(state) == list(range(num))


def test_load_run(tmp_path):
    # Two replicas swap states at iteration 2; the series of each state must follow the state.
    history = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    U = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    np.savez(
        tmp_path / "samples.npz", protocol="C(1,0) D(11,1)[0.25] E(2,0)", lambdas=np.array([0.0, 1.0]),
        H=np.array([[0, 0, 0], [1, 1, 1]], float), temperature=300.0, iteration_ps=1.0, equilibration=1,
        exchange=True, state_of_replica_history=history, U=U, accepted=np.zeros((2, 2)), attempted=np.zeros((2, 2)),
    )
    run = load_run(str(tmp_path))
    assert run.temperature == 300.0
    assert run.capping_orders == (3, 3)  # the default when a run predates the option
    assert np.array_equal(run.series[0], np.array([U[1, 0], U[2, 1], U[3, 1]]))
    assert np.array_equal(run.series[1], np.array([U[1, 1], U[2, 0], U[3, 0]]))
