"""Tests for structural operations: add/remove units and connections,
connect_to_output, add_layer, generation, and pruning.

All tests verify structural changes through observable forward-pass output.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from plastax import Network, StructureUpdateState, tree_replace

from conftest import (
    SmallHiddenNeuron,
    SmallOutputNeuron,
    constant_weight_init,
    forward_output,
    make_forward_only_fns,
)


# ---------------------------------------------------------------------------
# auto_connect_to_output on init
# ---------------------------------------------------------------------------

def test_auto_connect_to_output_on_init(key: PRNGKeyArray):
    """At init, auto_connect_to_output=True wires outputs to all inputs with
    init weights; output is the weighted sum. When False, output has no
    connections so output is 0. Test asserts outcome only (output values)."""
    fns = make_forward_only_fns(state_init_fn=constant_weight_init(1.0))
    inputs = jnp.array([2.0, 3.0])
    key, k_true, k_false = jax.random.split(key, 3)

    net_true = Network(
        n_inputs=2,
        n_outputs=1,
        max_hidden_per_layer=4,
        max_layers=2,
        hidden_neuron_cls=SmallHiddenNeuron,
        output_neuron_cls=SmallOutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
        auto_connect_to_output=True,
        key=k_true,
    )
    out_true = forward_output(net_true, inputs, k_true)
    # Hand-calc: output connected to input0 and input1 with weight 1.0 each
    # → output = 1.0*2 + 1.0*3 = 5.0
    assert jnp.allclose(out_true, jnp.array([5.0]), atol=1e-5)

    net_false = Network(
        n_inputs=2,
        n_outputs=1,
        max_hidden_per_layer=4,
        max_layers=2,
        hidden_neuron_cls=SmallHiddenNeuron,
        output_neuron_cls=SmallOutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
        auto_connect_to_output=False,
        key=k_false,
    )
    out_false = forward_output(net_false, inputs, k_false)
    assert jnp.allclose(out_false, jnp.array([0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# connect_to_output
# ---------------------------------------------------------------------------

def test_connect_to_output(empty_net: Network, key: PRNGKeyArray):
    """connect_to_output wires a hidden neuron to the output, and the output
    reflects that neuron's activation."""
    net = empty_net

    # Add one hidden neuron in layer 0 connected to both inputs
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    key, k1, k2, kf = jax.random.split(key, 4)
    net, success = net.add_unit(incoming, k1, connect_to_output=False)
    assert success

    # Wire it to the output
    abs_idx = net.get_units_in_layer(0)[0]  # first slot in layer 0
    net = net.connect_to_output(abs_idx[None], k2)

    # identity forward: hidden = 1.0*2 + 1.0*3 = 5.0, output = 1.0*5.0 = 5.0
    out = forward_output(net, jnp.array([2.0, 3.0]), kf)
    assert jnp.allclose(out, jnp.array([5.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# add_unit
# ---------------------------------------------------------------------------

def test_add_unit_from_inputs(empty_net: Network, key: PRNGKeyArray):
    """add_unit creates a neuron in layer 0 from input IDs and it contributes
    to the output when connect_to_output=True."""
    net = empty_net
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    key, k1, kf = jax.random.split(key, 3)
    net, success = net.add_unit(incoming, k1, connect_to_output=True)
    assert success

    # identity activation: hidden = 1*2 + 1*3 = 5, output = 1*5 = 5
    out = forward_output(net, jnp.array([2.0, 3.0]), kf)
    assert jnp.allclose(out, jnp.array([5.0]), atol=1e-5)


def test_add_unit_from_hidden(empty_net: Network, key: PRNGKeyArray):
    """add_unit places a neuron in layer 1 when incoming IDs reference a
    layer-0 neuron. The chain input -> L0 -> L1 -> output works."""
    net = empty_net

    # Add neuron A in layer 0
    key, k1, k2, kf = jax.random.split(key, 4)
    incoming_a = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming_a, k1, connect_to_output=True)
    abs_a = net.get_units_in_layer(0)[0]

    # Add neuron B in layer 1 (incoming from A)
    incoming_b = jnp.array([abs_a, -1, -1, -1], dtype=jnp.int32)
    net, success = net.add_unit(incoming_b, k2, connect_to_output=True)
    assert success

    # A = 1*2 + 1*3 = 5. B = 1*5 = 5. Output = 1*A + 1*B = 10.
    out = forward_output(net, jnp.array([2.0, 3.0]), kf)
    assert jnp.allclose(out, jnp.array([10.0]), atol=1e-5)


def test_add_unit_returns_false_when_full(empty_net: Network, key: PRNGKeyArray):
    """add_unit returns success=False when the target layer is full."""
    net = empty_net
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)

    # Fill all 4 slots in layer 0
    for i in range(4):
        key, ki = jax.random.split(key)
        net, success = net.add_unit(incoming, ki)
        assert success

    # 5th should fail
    key, ki = jax.random.split(key)
    net, success = net.add_unit(incoming, ki)
    assert not success


# ---------------------------------------------------------------------------
# remove_unit
# ---------------------------------------------------------------------------

def test_remove_unit(empty_net: Network, key: PRNGKeyArray):
    """remove_unit removes a neuron's contribution from the output."""
    net = empty_net
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)

    # Add two hidden neurons, both connected to output
    key, k1, k2, kf1, kf2 = jax.random.split(key, 5)
    net, _ = net.add_unit(incoming, k1, connect_to_output=True)
    net, _ = net.add_unit(incoming, k2, connect_to_output=True)

    # Both contribute: output = 5 + 5 = 10
    out_before = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_before, jnp.array([10.0]), atol=1e-5)

    # Remove the first neuron
    abs_first = net.get_units_in_layer(0)[0]
    net = net.remove_unit(abs_first)

    # Only second contributes: output = 5
    out_after = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_after, jnp.array([5.0]), atol=1e-5)


def test_remove_unit_cleans_hidden_connections(empty_net: Network, key: PRNGKeyArray):
    """remove_unit deactivates connections in downstream neurons that pointed
    to the removed neuron."""
    net = empty_net

    # A in layer 0
    key, k1, k2, kf1, kf2 = jax.random.split(key, 5)
    incoming_a = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming_a, k1)
    abs_a = net.get_units_in_layer(0)[0]

    # B in layer 1, connected only to A, wired to output
    incoming_b = jnp.array([abs_a, -1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming_b, k2, connect_to_output=True)

    # Before removal: B = A = 5, output = 5
    out_before = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_before, jnp.array([5.0]), atol=1e-5)

    # Remove A — B's connection to A should be deactivated
    net = net.remove_unit(abs_a)

    # B has no active connections -> activation 0, output = 0
    out_after = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_after, jnp.array([0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# add_connection / remove_connection
# ---------------------------------------------------------------------------

def test_add_connection_to_hidden(empty_net: Network, key: PRNGKeyArray):
    """Adding a connection to a hidden neuron changes its forward activation."""
    net = empty_net

    # Add hidden neuron connected only to input 0
    key, k1, k2, kf1, kf2 = jax.random.split(key, 5)
    incoming = jnp.array([0, -1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming, k1, connect_to_output=True)

    # output = relu(1*in0) = 2
    out_before = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_before, jnp.array([2.0]), atol=1e-5)

    # Add connection from input 1 (abs idx 1) to hidden neuron 0 (hidden-rel 0)
    net, success = net.add_connection_to_hidden(
        from_idx=jnp.array(1), to_idx=jnp.array(0), key=k2)
    assert success

    # The new weight is initialized by state_init_fn (constant 1.0)
    # output = 1*in0 + 1*in1 = 5
    out_after = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_after, jnp.array([5.0]), atol=1e-5)


def test_add_connection_to_output(empty_net: Network, key: PRNGKeyArray):
    """Adding a connection to an output neuron changes its forward value."""
    net = empty_net

    # Add hidden neuron connected to both inputs, wired to output
    key, k1, k2, kf1, kf2 = jax.random.split(key, 5)
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming, k1, connect_to_output=True)

    # output = hidden = 5
    out_before = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_before, jnp.array([5.0]), atol=1e-5)

    # Add direct connection from input 0 (abs 0) to output 0 (output-rel 0)
    net, success = net.add_connection_to_output(
        from_idx=jnp.array(0), to_idx=jnp.array(0), key=k2)
    assert success

    # output = hidden + input_0 = 5 + 2 = 7
    out_after = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_after, jnp.array([7.0]), atol=1e-5)


def test_remove_connection_from_hidden(empty_net: Network, key: PRNGKeyArray):
    """Removing a connection from a hidden neuron reduces its inputs."""
    net = empty_net

    # Add hidden neuron connected to inputs 0 and 1
    key, k1, kf1, kf2 = jax.random.split(key, 4)
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming, k1, connect_to_output=True)

    # output = 1*2 + 1*3 = 5
    out_before = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_before, jnp.array([5.0]), atol=1e-5)

    # Remove connection at slot 1 (input 1) from hidden neuron 0 (hidden-rel)
    net = net.remove_connection_from_hidden(
        neuron_idx=jnp.array(0), connection_slot=1)

    # output = 1*2 = 2
    out_after = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_after, jnp.array([2.0]), atol=1e-5)


def test_remove_connection_from_output(empty_net: Network, key: PRNGKeyArray):
    """Removing an output connection removes that source's contribution."""
    net = empty_net

    # Two hidden neurons, both connected to output
    key, k1, k2, kf1, kf2 = jax.random.split(key, 5)
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming, k1, connect_to_output=True)
    net, _ = net.add_unit(incoming, k2, connect_to_output=True)

    # output = 5 + 5 = 10
    out_before = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_before, jnp.array([10.0]), atol=1e-5)

    # Find which slot in the output connects to the first hidden neuron
    abs_first = int(net.get_units_in_layer(0)[0])
    output_ids = net.output_states.incoming_ids[0]
    output_mask = net.output_states.active_connection_mask[0]
    slot = None
    for s in range(output_ids.shape[0]):
        if output_mask[s] and int(output_ids[s]) == abs_first:
            slot = s
            break
    assert slot is not None

    net = net.remove_connection_from_output(
        neuron_idx=jnp.array(0), connection_slot=slot)

    # output = 5 (only second neuron)
    out_after = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_after, jnp.array([5.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# add_layer
# ---------------------------------------------------------------------------

def test_add_layer(empty_net: Network, key: PRNGKeyArray):
    """add_layer creates a full layer whose neurons contribute to the output."""
    net = empty_net
    key, k1, kf = jax.random.split(key, 3)
    net = net.add_layer(n_units=2, key=k1, connect_to_output=True)

    # Both neurons connect to inputs 0 and 1 (deterministic_connector, weight=1.0)
    # hidden_0 = 1*2 + 1*3 = 5, hidden_1 = 5. output = 5 + 5 = 10
    out = forward_output(net, jnp.array([2.0, 3.0]), kf)
    assert jnp.allclose(out, jnp.array([10.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Generation (via structure_update)
# ---------------------------------------------------------------------------

def _always_generate_one(layer_states, next_layer_states, structure_state):
    """Structure fn: always generate 1 neuron, never prune."""
    layer_size = layer_states.active_mask.shape[0]
    return structure_state, jnp.zeros(layer_size, dtype=bool), jnp.array(1, dtype=jnp.int32)


def test_generation_adds_neuron(key: PRNGKeyArray):
    """Structure update generation creates a neuron that participates in the
    forward pass."""
    fns = make_forward_only_fns(
        state_init_fn=constant_weight_init(0.5),
        output_state_init_fn=constant_weight_init(1.0),
        structure_update_fn=_always_generate_one,
    )
    # Use max_layers=1 so _structure_update only iterates one layer
    net = Network(
        n_inputs=2,
        n_outputs=1,
        max_hidden_per_layer=2,
        max_layers=1,
        hidden_neuron_cls=SmallHiddenNeuron,
        output_neuron_cls=SmallOutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
        max_generate_per_step=1,
        auto_connect_to_output=True,
    )

    # No hidden neurons yet, output should be sum of inputs
    kf1, kf2, kf3, kf4 = jax.random.split(key, 4)
    targets = jnp.zeros(net.n_outputs)
    inputs = jnp.array([2.0, 3.0])

    net = net.step(inputs, targets, kf1)
    assert jnp.allclose(net.output_states.activation_value, jnp.array([5.0]), atol=1e-5)

    # There should have been one hidden unit added after the last forward pass
    net = net.step(inputs, targets, kf2)
    assert jnp.allclose(net.output_states.activation_value, jnp.array([7.5]), atol=1e-5)

    # After one more generation there should be two hidden units, both in the same layer
    net = net.step(inputs, targets, kf3)
    assert jnp.allclose(net.output_states.activation_value, jnp.array([10.0]), atol=1e-5)

    # The maximum number of hidden units is 2, so no more hidden units should have been added
    net = net.step(inputs, targets, kf4)
    assert jnp.allclose(net.output_states.activation_value, jnp.array([10.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Pruning (via structure_update)
# ---------------------------------------------------------------------------

def _always_prune_all(layer_states, next_layer_states, structure_state):
    """Structure fn: prune all active neurons, generate none."""
    return structure_state, layer_states.active_mask, jnp.array(0, dtype=jnp.int32)


def test_pruning_removes_neuron(key: PRNGKeyArray):
    """Structure update pruning deactivates a neuron so it no longer
    contributes to the output."""
    fns = make_forward_only_fns(structure_update_fn=_always_prune_all)
    net = Network(
        n_inputs=2,
        n_outputs=1,
        max_hidden_per_layer=4,
        max_layers=2,
        hidden_neuron_cls=SmallHiddenNeuron,
        output_neuron_cls=SmallOutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
    )

    # Add a hidden neuron connected to output
    key, k1, kf1, k_step, kf2 = jax.random.split(key, 5)
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    net, _ = net.add_unit(incoming, k1, connect_to_output=True)

    out_before = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_before, jnp.array([5.0]), atol=1e-5)

    # Step triggers pruning
    net = net.step(jnp.array([2.0, 3.0]), jnp.array([0.0]), k_step)

    # Neuron pruned, output = 0
    out_after = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_after, jnp.array([0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Prune then generate
# ---------------------------------------------------------------------------

def _prune_all_generate_one(layer_states, next_layer_states, structure_state):
    """Structure fn: prune all active neurons, then generate 1."""
    return structure_state, layer_states.active_mask, jnp.array(1, dtype=jnp.int32)


def test_prune_then_generate_reuses_slot(key: PRNGKeyArray):
    """After pruning, a newly generated neuron gets fresh weights, not the old ones."""
    fns = make_forward_only_fns(
        state_init_fn=constant_weight_init(0.5),
        output_state_init_fn=constant_weight_init(1.0),
        structure_update_fn=_prune_all_generate_one,
    )
    # Use max_layers=1 so _structure_update only iterates one layer
    net = Network(
        n_inputs=2,
        n_outputs=1,
        max_hidden_per_layer=4,
        max_layers=1,
        hidden_neuron_cls=SmallHiddenNeuron,
        output_neuron_cls=SmallOutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
        max_generate_per_step=1,
        auto_connect_to_output=True,
    )

    # Seed with a neuron that has weight=1.0 via add_unit (uses the fns'
    # state_init_fn which is constant 0.5, but we override the weights manually)
    key, k1, kf1, k_step, kf2 = jax.random.split(key, 5)
    incoming_ids = jnp.array([0, 1], dtype=jnp.int32)
    net, success = net.add_unit(incoming_ids, k1, connect_to_output=True)
    
    assert success, "Failed to add unit"

    # Manually set this neuron's weights to 1.0 so we can distinguish old from new
    net = eqx.tree_at(
        lambda n: n.hidden_states.weights,
        net,
        net.hidden_states.weights.at[0].set(1.0),
    )

    # With weight=1.0: inputs contribute 5, hidden unit contributes 5, output = 10
    out_old = forward_output(net, jnp.array([2.0, 3.0]), kf1)
    assert jnp.allclose(out_old, jnp.array([10.0]), atol=1e-5)

    # Step: prunes old neuron, generates new one with weight=0.5
    net = net.step(jnp.array([2.0, 3.0]), jnp.array([0.0]), k_step)

    # With weight=0.5: inputs contribute 5, hidden unit contributes 2.5, output = 7.5
    out_new = forward_output(net, jnp.array([2.0, 3.0]), kf2)
    assert jnp.allclose(out_new, jnp.array([7.5]), atol=1e-5)
