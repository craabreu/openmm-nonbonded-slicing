import itertools

import numpy as np
import pytest

from lbf import (
    EXP,
    LEGACY_TERMS,
    SMOOTH,
    TERMS,
    PaperProtocol,
    Protocol,
    bernstein,
    closest_contact,
    coefficients,
    energies_in_basis,
    equal_length_lambdas,
    exponential,
    feedback_lambdas,
    fit_lambda_of_c,
    in_basis,
    lambda_of_c,
    load_run,
    mix_replicas,
    nonincreasing_fit,
    replica_flow,
    smooth_step,
    states_in_basis,
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


def test_exponential():
    x = np.linspace(0, 1, 101)
    s, ds = exponential(2.5, x)
    assert s[0] == 0 and s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(s) > 0)
    assert ds[0] == pytest.approx(2.5 / np.expm1(2.5)) and ds[-1] == pytest.approx(2.5 * np.exp(2.5) / np.expm1(2.5))
    assert np.allclose(ds, finite_difference(lambda t: exponential(2.5, t)[0], x), rtol=1e-6)
    with pytest.raises(ValueError):
        exponential(0.0, x)


def test_exponential_protocol():
    protocol = Protocol.parse("C(exp 2.5) D(7,2)[0.4] E(2,1)[0.4] Q=E")
    assert protocol.specs[0] == (EXP, 2.5, 0.0)
    lam = np.linspace(0, 1, 11)
    h, dh = protocol.h(lam)
    assert np.allclose(h[0], exponential(2.5, lam)[0]) and np.allclose(dh[0], exponential(2.5, lam)[1])
    assert protocol.end_orders() == (0, 0)
    assert protocol.charging_hazard() == pytest.approx(0.25, abs=0.01)
    with pytest.raises(ValueError):
        Protocol.parse("C(exp 2.5,1) D(7,2)[0.4] E(2,1)[0.4] Q=E")


@pytest.mark.parametrize("text", ["C(1,0) D(11,1)[0.25] E(2,0)", "C(0,2) D(inf)[0.25] E(5,1)",
                                  "C(2,1) D(7,2)[0.4] E(2,1)[0.4] Q=E", "C(2,1) D(7,2)[0.4] E(4,1) Q(9,1)[0.5]",
                                  "C(exp 2.5) D(7,2)[0.4] E(2,1)[0.4] Q=E", "C(2,1) D(exp 3)[0.4] E(exp 1.5)[0.4]"])
def test_protocol_text(text):
    protocol = Protocol.parse(text)
    assert str(protocol) == text
    assert Protocol.parse(str(protocol)) == protocol
    assert Protocol.parse(text.replace(" ", "")) == protocol


def test_coulomb_decapping_ties():
    C, D, E = (2, 1, 0.0), (7, 2, 0.4), (4, 1, 0.0)
    assert Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1)").specs == (C, D, D, E)
    assert Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1) Q=D").specs == (C, D, D, E)
    assert Protocol.parse("C(2,1) D(7,2)[0.4] Q=E E(4,1)").specs == (C, D, E, E)
    assert str(Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1) Q=D")) == "C(2,1) D(7,2)[0.4] E(4,1)"
    assert str(Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1) Q(7,2)[0.4]")) == "C(2,1) D(7,2)[0.4] E(4,1)"
    assert str(Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1) Q(4,1)")) == "C(2,1) D(7,2)[0.4] E(4,1) Q=E"


@pytest.mark.parametrize("text", ["C(1,0) D(11,1)", "C(1,0) D(11,1) E(2,0) X", "C(1,0) D(11) E(2,x)",
                                  "C(1,0) D(11,1) E(2,0) Q=E Q=D", "C(1,0) D(11,1) E(2,0) Q=E Q(2,0)", "Q=E"])
def test_protocol_errors(text):
    with pytest.raises(ValueError):
        Protocol.parse(text)


@pytest.mark.parametrize("protocol", [Protocol.parse("C(1,0) D(11,1)[0.25] E(2,0)"),
                                      Protocol.parse("C(1,1) D(inf)[0.25] E(2,1)[0.4] Q=E"), PaperProtocol()])
def test_protocol_h(protocol):
    lam = np.linspace(0, 1, 201)
    h, dh = protocol.h(lam)
    assert h.shape == dh.shape == (len(TERMS), len(lam))
    assert np.allclose(h[:, 0], 0) and np.allclose(h[:, -1], 1)
    smooth = (lam > 0.01) & (lam < 0.99) & np.all(np.abs(lam[:, None] - [0.25, 0.4, 0.6, 0.7, 0.8]) > 0.01, axis=1)
    fd = finite_difference(lambda x: protocol.h(x)[0], lam)
    assert np.allclose(dh[:, smooth], fd[:, smooth], rtol=1e-5, atol=1e-7)


def test_charging_hazard():
    assert Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1)").charging_hazard() == pytest.approx(0.44, abs=0.01)
    assert Protocol.parse("C(2,1) D(7,2)[0.4] E(2,1)[0.4] Q=E").charging_hazard() == pytest.approx(0.16, abs=0.01)
    assert Protocol.parse("C(2,1) D(7,2)[0.4] E(2,1)").charging_hazard() == 1.0  # h_E = h_C
    assert PaperProtocol().charging_hazard() == 0.0


def test_paper_protocol_is_bare_coulomb():
    h, dh = PaperProtocol().h(np.linspace(0, 1, 101))
    assert np.array_equal(h[2], h[3]) and np.array_equal(dh[2], dh[3])
    assert not PaperProtocol().representable_in(LEGACY_TERMS)


def test_window():
    h, dh = Protocol.parse("C(1,0) D(11,1)[0.25] E(2,0)").h(np.linspace(0, 1, 101))
    lam = np.linspace(0, 1, 101)
    assert np.all(h[1, lam <= 0.25] == 0) and np.all(dh[1, lam <= 0.25] == 0)
    assert np.all(h[1, lam > 0.25] > 0)


def test_bases():
    lam = np.linspace(0, 1, 11)
    capped = Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1)")
    bare = Protocol.parse("C(2,1) D(7,2)[0.4] E(4,1) Q=E")
    h, dh = capped.h(lam)
    assert in_basis(TERMS, h, dh) == (h, dh)
    h3, dh3 = in_basis(LEGACY_TERMS, h, dh)
    assert h3.shape == (3, len(lam)) and np.array_equal(h3, h[[0, 1, 3]]) and np.array_equal(dh3, dh[[0, 1, 3]])
    assert capped.representable_in(LEGACY_TERMS) and bare.representable_in(TERMS)
    assert not bare.representable_in(LEGACY_TERMS)
    with pytest.raises(ValueError):
        in_basis(LEGACY_TERMS, *bare.h(lam))
    U = np.arange(12.0).reshape(3, 4)
    assert energies_in_basis(TERMS, U, TERMS) is U
    assert np.array_equal(energies_in_basis(LEGACY_TERMS, U, TERMS), np.stack([U[:, 0], U[:, 1] + U[:, 2], U[:, 3]], 1))
    with pytest.raises(ValueError):
        energies_in_basis(TERMS, U[:, :3], LEGACY_TERMS)
    # the states of a four-term run join a legacy pool only if their h_Q equals their h_D
    H = capped.h(lam)[0].T
    assert np.array_equal(states_in_basis(LEGACY_TERMS, H, TERMS), H[:, [0, 1, 3]])
    with pytest.raises(ValueError):
        states_in_basis(LEGACY_TERMS, bare.h(lam)[0].T, TERMS)


def test_equal_length_lambdas():
    lam = np.linspace(0, 1, 1001)
    assert np.allclose(equal_length_lambdas(lam, np.ones_like(lam), 11), np.linspace(0, 1, 11))
    # sd = 2 lambda: the length up to lambda is lambda^2
    assert np.allclose(equal_length_lambdas(lam, 2 * lam, 6), np.sqrt(np.linspace(0, 1, 6)), atol=1e-4)
    # no length at all over the first half: every state but the first lies in the second half
    places = equal_length_lambdas(lam, (lam > 0.5).astype(float), 5)
    assert places[0] == 0 and np.all(places[1:] > 0.5) and places[-1] == 1


def test_replica_flow():
    # replica 0 walks 0 -> 1 -> 2 -> 1 -> 0 -> 1 -> 2 (one round trip), replica 1 sits at state 1 unlabeled,
    # and replica 2 sits at state 2 as a down-walker from the start
    history = np.array([[0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 1, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2]])
    f, visits, trips = replica_flow(history)
    assert trips == 1
    assert visits.tolist() == [2, 3, 9]  # the unlabeled replica is not counted
    assert np.allclose(f, [1.0, 2 / 3, 0.0])  # state 1: up at iterations 1 and 5, down at 3


def test_nonincreasing_fit():
    assert nonincreasing_fit([3, 2, 1]).tolist() == [3, 2, 1]
    assert nonincreasing_fit([1, 3, 2]).tolist() == [2, 2, 2]
    assert nonincreasing_fit([5, 1, 2, 0]).tolist() == [5, 1.5, 1.5, 0]
    assert nonincreasing_fit([0, 1], weights=[3, 1]).tolist() == [0.25, 0.25]


def test_feedback_lambdas():
    lam = np.linspace(0, 1, 11)
    # a linear f gives uniform states; f steeper on the left gives denser states there
    assert np.allclose(feedback_lambdas(lam, 1 - lam, 6), np.linspace(0, 1, 6), atol=1e-9)
    states = feedback_lambdas(lam, 1 - np.sqrt(lam), 6)
    assert states[0] == 0 and states[-1] == 1 and np.all(np.diff(states) > 0) and states[1] < 0.2
    with pytest.raises(ValueError):
        feedback_lambdas(lam, np.ones(11), 6)


def test_closest_contact():
    box = np.array([3.0, 3.0, 3.0])
    positions = np.array([[0.1, 0.1, 0.1], [1.5, 1.5, 1.5], [2.9, 0.1, 0.1], [1.0, 1.0, 1.0]])
    # the polar atom 0 is 0.2 nm from atom 2 through the periodic boundary, and 1.56 nm from atom 3
    assert closest_contact(positions, box, np.array([0]), np.array([2, 3])) == pytest.approx(0.2)
    assert closest_contact(positions, box, np.array([1]), np.array([3])) == pytest.approx(np.sqrt(3 * 0.25))



def test_collapse_report():
    from simulate import collapse_report

    # three states: decoupled, charged with weak decapping, and fully coupled; two replicas that swap
    H = np.array([[0.0, 0.0, 0.0, 0.0], [0.2, 1e-5, 0.02, 0.02], [1.0, 1.0, 1.0, 1.0]])
    history = np.array([[0, 1, 2], [1, 0, 2]] * 50)
    contact = np.full(history.shape, 0.28)
    contact[:, 0] = 0.05  # the replica in state 0 or 1 overlaps every other frame, which is normal at state 0
    report = collapse_report(contact, history, H)
    assert "collapse in states [1]" in report
    contact[history == 1] = np.where(np.arange(100) % 25 == 0, 0.15, 0.28)  # 4% of state 1: brief overlaps
    assert "collapse" not in collapse_report(contact, history, H).splitlines()[-1]

def test_end_orders():
    assert Protocol.parse("C(2,1) D(7,2)[0.4] E(2,1)[0.4] Q=E").end_orders() == (2, 1)
    assert Protocol.parse("C(3,2) D(inf)[0.4] E(1,1)").end_orders() == (1, 1)
    assert PaperProtocol().end_orders() == (2, 2)


def test_lambda_of_c():
    x = np.linspace(0, 1, 7)
    assert np.allclose(bernstein(x, 3).sum(axis=1), 1) and np.allclose(bernstein(x, 3)[:, 0], (1 - x) ** 3)
    c = np.linspace(0, 1, 1001)
    w = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    lam, dlam = lambda_of_c(c, 2, 1, w)
    assert lam[0] == 0 and lam[-1] == 1 and np.all(np.diff(lam) > 0) and np.all(dlam[1:-1] > 0)
    assert np.allclose(dlam[1:-1], finite_difference(lambda y: lambda_of_c(y, 2, 1, w)[0], c[1:-1]), rtol=1e-4)
    # the end exponents: lambda ~ c^(1/3) at 0 and 1 - lambda ~ (1 - c)^(1/2) at 1
    small = np.array([1e-6, 1e-5])
    assert np.allclose(np.diff(np.log(lambda_of_c(small, 2, 1, w)[0])) / np.log(10), 1 / 3, atol=1e-3)
    assert np.allclose(np.diff(np.log(1 - lambda_of_c(1 - small, 2, 1, w)[0])) / np.log(10), 1 / 2, atol=1e-3)


def test_fit_lambda_of_c():
    # sd proportional to dS_{2,1}/dlambda: the exact c is S_{2,1}(lambda), with end orders (2, 1)
    lam = np.linspace(0, 1, 2001)
    sd = 3.0 * switch(2, 1, lam)[1]
    w, flatness = fit_lambda_of_c(lam, sd, 2, 1, 6, nodes=1001)
    assert w[0] == 0 and np.isclose(w[-1], 1) and np.all(np.diff(w) >= 0)
    assert flatness < 1.002
    c = np.linspace(0, 1, 101)
    assert np.allclose(switch(2, 1, lambda_of_c(c, 2, 1, w)[0])[0], c, atol=0.01)


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
    assert run.terms == LEGACY_TERMS  # and the single decapping energy when it predates the split
    assert np.array_equal(run.series[0], np.array([U[1, 0], U[2, 1], U[3, 1]]))
    assert np.array_equal(run.series[1], np.array([U[1, 1], U[2, 0], U[3, 0]]))
    np.savez(
        tmp_path / "samples.npz", protocol="C(1,0) D(11,1)[0.25] E(2,0) Q=E", lambdas=np.array([0.0, 1.0]),
        H=np.zeros((2, 4)), terms=np.array(TERMS), temperature=300.0, iteration_ps=1.0, equilibration=1,
        exchange=True, state_of_replica_history=history, U=np.arange(4 * 2 * 4, dtype=float).reshape(4, 2, 4),
        accepted=np.zeros((2, 2)), attempted=np.zeros((2, 2)), capping_orders=np.array([3, 3]),
    )
    run = load_run(str(tmp_path))
    assert run.terms == TERMS and run.series[0].shape == (3, 4)
