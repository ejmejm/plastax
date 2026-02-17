from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from plastax.states import (
    BackwardPassState,
    ConnectivityState,
    NeuronState,
    StructureUpdateState,
    tree_replace,
)


# ---------------------------------------------------------------------------
# State update function container
# ---------------------------------------------------------------------------

class StateUpdateFunctions(eqx.Module):
    forward_fn: Callable = eqx.field(static=True)
    backward_signal_fn: Callable = eqx.field(static=True)
    neuron_update_fn: Callable = eqx.field(static=True)
    structure_update_fn: Callable = eqx.field(static=True)
    init_neuron_fn: Callable = eqx.field(static=True)
    compute_output_error_fn: Callable = eqx.field(static=True)
    output_forward_fn: Callable | None = eqx.field(static=True, default=None)
    output_neuron_update_fn: Callable | None = eqx.field(static=True, default=None)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network(eqx.Module):
    # Static (config)
    n_inputs: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    max_hidden_per_layer: int = eqx.field(static=True)
    max_layers: int = eqx.field(static=True)
    max_connections: int = eqx.field(static=True)
    max_output_connections: int = eqx.field(static=True)
    max_generate_per_step: int = eqx.field(static=True)
    auto_connect_to_output: bool = eqx.field(static=True)
    layer_boundaries: tuple = eqx.field(static=True)
    total_hidden: int = eqx.field(static=True)
    total_neurons: int = eqx.field(static=True)
    state_update_fns: StateUpdateFunctions = eqx.field(static=True)

    # Dynamic (state)
    input_values: Float[Array, 'n_inputs']
    hidden_states: NeuronState
    output_states: NeuronState
    structure_state: StructureUpdateState

    def __init__(
        self,
        *,
        n_inputs: int,
        n_outputs: int,
        max_hidden_per_layer: int,
        max_layers: int,
        hidden_neuron_cls: type[NeuronState] = NeuronState,
        output_neuron_cls: type[NeuronState] = None,
        state_update_fns: StateUpdateFunctions,
        structure_state: StructureUpdateState,
        max_generate_per_step: int = 0,
        auto_connect_to_output: bool = False,
    ):
        """Create a Network with inputs directly connected to outputs, all hidden inactive.

        Uses the given NeuronState classes (or subclasses) to construct the
        initial state arrays.  max_connections and max_output_connections are
        derived from the constructed neurons' connectivity shapes.  Output
        neurons are activated and wired to receive from all input units.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_hidden_per_layer = max_hidden_per_layer
        self.max_layers = max_layers
        self.max_generate_per_step = max_generate_per_step
        self.auto_connect_to_output = auto_connect_to_output
        self.state_update_fns = state_update_fns

        total_hidden = max_hidden_per_layer * max_layers
        self.total_hidden = total_hidden
        self.total_neurons = n_inputs + total_hidden + n_outputs
        self.layer_boundaries = tuple(
            (k * max_hidden_per_layer, (k + 1) * max_hidden_per_layer)
            for k in range(max_layers)
        )

        # Input activations default to zeros
        self.input_values = jnp.zeros(n_inputs)

        # If output_neuron_cls is None, use hidden_neuron_cls
        if output_neuron_cls is None:
            output_neuron_cls = hidden_neuron_cls

        # Create initial states by vmapping over constructors
        hidden_states = jax.vmap(lambda _: hidden_neuron_cls())(jnp.arange(total_hidden))
        output_states = jax.vmap(lambda _: output_neuron_cls())(jnp.arange(n_outputs))
        self.max_connections = hidden_states.connectivity.weights.shape[-1]
        self.max_output_connections = output_states.connectivity.weights.shape[-1]

        # Activate output neurons
        output_states = eqx.tree_at(
            lambda s: s.active_mask, output_states,
            jnp.ones(n_outputs, dtype=bool),
        )

        # Optionally connect output neurons to input neurons
        if auto_connect_to_output:
            input_ids = jnp.arange(n_inputs, dtype=jnp.int32)
            conn = output_states.connectivity
            conn = tree_replace(
                conn,
                incoming_ids=conn.incoming_ids.at[:, :n_inputs].set(input_ids),
                weights=conn.weights.at[:, :n_inputs].set(0.0),
                active_connection_mask=conn.active_connection_mask.at[:, :n_inputs].set(True),
            )
            output_states = eqx.tree_at(
                lambda s: s.connectivity,
                output_states,
                conn,
            )

        self.hidden_states = hidden_states
        self.output_states = output_states
        self.structure_state = structure_state

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def step(
        self,
        inputs: Float[Array, 'n_inputs'],
        targets: Float[Array, 'n_outputs'],
        key: PRNGKeyArray,
    ) -> 'Network':
        """Run one full step: forward -> backward -> structure update -> generation."""
        fns = self.state_update_fns
        network = tree_replace(self, input_values=inputs)
        network = network._forward_pass(fns.forward_fn, fns.output_forward_fn)
        network = network._backward_pass(targets, fns)
        network, generation_specs = network._structure_update(fns.structure_update_fn)
        network = network._generate_neurons(generation_specs, fns.init_neuron_fn, key)
        return network

    # -----------------------------------------------------------------------
    # Ops (public)
    # -----------------------------------------------------------------------

    def add_unit(
        self, neuron_state: NeuronState, connect_to_output: bool = False,
    ) -> Tuple['Network', Bool[Array, '']]:
        """Insert a neuron into the correct hidden layer based on its incoming connections.

        The target layer is determined automatically: if the neuron's active
        incoming connections reference layer *k* (or the inputs), the neuron is
        placed in layer *k+1* (or layer 0).

        Returns the updated network and a boolean indicating success (False if
        the target layer had no inactive slots).
        """
        # Determine target layer from active incoming connections
        active_ids = jnp.where(
            neuron_state.connectivity.active_connection_mask,
            neuron_state.connectivity.incoming_ids, -1,
        )
        max_id = jnp.max(active_ids)

        layer_idx = jnp.where(
            max_id < self.n_inputs, 0,
            (max_id - self.n_inputs) // self.max_hidden_per_layer + 1,
        )
        start = layer_idx * self.max_hidden_per_layer

        # Find first inactive slot in the target layer
        layer_active = jax.lax.dynamic_slice(
            self.hidden_states.active_mask, (start,), (self.max_hidden_per_layer,),
        )
        slot = jnp.argmin(layer_active)
        has_slot = ~layer_active[slot]
        hidden_rel = start + slot

        # Insert neuron at the slot
        hidden_states = jax.tree.map(
            lambda full, single: jnp.where(has_slot, full.at[hidden_rel].set(single), full),
            self.hidden_states, neuron_state,
        )

        output_states = self.output_states
        if connect_to_output:
            abs_index = hidden_rel + self.n_inputs
            output_states = self._auto_connect_to_output(output_states, abs_index, has_slot)

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states), has_slot

    def remove_unit(self, neuron_abs_idx: Int[Array, '']) -> 'Network':
        """Deactivate a hidden neuron and disconnect other neurons from it.

        The neuron's own incoming connections are left as-is since the slot
        will be fully overwritten if reused by add_unit or _generate_neurons.
        Connections from other neurons TO this one must be cleaned up so they
        don't accidentally attach to a future occupant of the same slot.
        """
        hidden_states = self.hidden_states
        output_states = self.output_states
        hidden_rel = neuron_abs_idx - self.n_inputs

        # Deactivate the neuron
        hidden_states = eqx.tree_at(
            lambda s: s.active_mask, hidden_states,
            hidden_states.active_mask.at[hidden_rel].set(False),
        )

        # Deactivate connections pointing to this neuron in hidden layers
        hidden_match = hidden_states.connectivity.incoming_ids == neuron_abs_idx
        hidden_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, hidden_states,
            jnp.where(hidden_match, False, hidden_states.connectivity.active_connection_mask),
        )

        # Deactivate connections pointing to this neuron in output layer
        output_match = output_states.connectivity.incoming_ids == neuron_abs_idx
        output_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, output_states,
            jnp.where(output_match, False, output_states.connectivity.active_connection_mask),
        )

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states)

    def add_connection_to_hidden(
        self,
        from_idx: Int[Array, ''],
        to_idx: Int[Array, ''],
        weight: Float[Array, ''] = jnp.array(0.0),
    ) -> Tuple['Network', Bool[Array, '']]:
        """Add a connection to a hidden neuron.

        Both indices are relative to the hidden state array.  Returns the
        updated network and a boolean indicating success (False if the neuron
        had no inactive connection slots).
        """
        hidden_states, success = self._add_connection(
            self.hidden_states, to_idx, from_idx, weight)
        return tree_replace(self, hidden_states=hidden_states), success

    def add_connection_to_output(
        self,
        from_idx: Int[Array, ''],
        to_idx: Int[Array, ''],
        weight: Float[Array, ''] = jnp.array(0.0),
    ) -> Tuple['Network', Bool[Array, '']]:
        """Add a connection to an output neuron.

        Both indices are relative to the output state array.  Returns the
        updated network and a boolean indicating success (False if the neuron
        had no inactive connection slots).
        """
        output_states, success = self._add_connection(
            self.output_states, to_idx, from_idx, weight)
        return tree_replace(self, output_states=output_states), success

    def remove_connection_from_hidden(
        self, neuron_idx: Int[Array, ''], connection_slot: int,
    ) -> 'Network':
        """Deactivate a connection at the given slot of a hidden neuron.

        neuron_idx is relative to the hidden state array.
        """
        hidden_states = self._remove_connection(self.hidden_states, neuron_idx, connection_slot)
        return tree_replace(self, hidden_states=hidden_states)

    def remove_connection_from_output(
        self, neuron_idx: Int[Array, ''], connection_slot: int,
    ) -> 'Network':
        """Deactivate a connection at the given slot of an output neuron.

        neuron_idx is relative to the output state array.
        """
        output_states = self._remove_connection(self.output_states, neuron_idx, connection_slot)
        return tree_replace(self, output_states=output_states)

    # -----------------------------------------------------------------------
    # Internal: forward / backward / structure / generation
    # -----------------------------------------------------------------------

    def _build_all_activations(self) -> Float[Array, 'total_neurons']:
        return jnp.concatenate([
            self.input_values,
            self.hidden_states.forward_state.activation_value,
            self.output_states.forward_state.activation_value,
        ])

    def _forward_pass(
        self, forward_fn: Callable, output_forward_fn: Callable | None = None,
    ) -> 'Network':
        """Run forward pass layer by layer (global mode)."""
        hidden_states = self.hidden_states
        all_activations = self._build_all_activations()

        for layer_k in range(self.max_layers):
            start, end = self.layer_boundaries[layer_k]
            layer_states = self._get_layer_states(hidden_states, start, end)

            incoming_acts = all_activations[layer_states.connectivity.incoming_ids]
            new_activations, updated_states = jax.vmap(forward_fn)(layer_states, incoming_acts)
            new_activations = new_activations * layer_states.active_mask

            hidden_states = self._set_layer_states(hidden_states, start, end, updated_states)
            abs_start = self.n_inputs + start
            abs_end = self.n_inputs + end
            all_activations = all_activations.at[abs_start:abs_end].set(new_activations)

        output_states = self.output_states
        output_incoming_acts = all_activations[output_states.connectivity.incoming_ids]
        out_fn = output_forward_fn if output_forward_fn is not None else forward_fn
        _, updated_output_states = jax.vmap(out_fn)(output_states, output_incoming_acts)

        return tree_replace(self, hidden_states=hidden_states, output_states=updated_output_states)

    def _backward_pass(
        self, targets: Float[Array, 'n_outputs'], user_fns: StateUpdateFunctions,
    ) -> 'Network':
        """Run backward pass: output error -> backward signal -> neuron update, layer by layer."""
        hidden_states = self.hidden_states
        output_states = self.output_states

        # 1. Compute output error
        output_activations = output_states.forward_state.activation_value
        output_error = user_fns.compute_output_error_fn(output_activations, targets)
        output_states = eqx.tree_at(
            lambda s: s.backward_state.error_signal, output_states, output_error)

        # 2. Save output pre-update weights, then run neuron update
        output_pre_update_weights = output_states.connectivity.weights
        out_update_fn = (user_fns.output_neuron_update_fn
                         if user_fns.output_neuron_update_fn is not None
                         else user_fns.neuron_update_fn)
        output_states = jax.vmap(out_update_fn)(output_states)

        # 3. Hidden layers from last to first
        for layer_k in range(self.max_layers - 1, -1, -1):
            start, end = self.layer_boundaries[layer_k]
            layer_states = self._get_layer_states(hidden_states, start, end)
            layer_indices = jnp.arange(start, end) + self.n_inputs

            # Next layer states with PRE-update weights
            if layer_k == self.max_layers - 1:
                next_layer_states = eqx.tree_at(
                    lambda s: s.connectivity.weights,
                    output_states, output_pre_update_weights,
                )
            else:
                next_start, next_end = self.layer_boundaries[layer_k + 1]
                next_layer_states = self._get_layer_states(hidden_states, next_start, next_end)
                next_layer_states = eqx.tree_at(
                    lambda s: s.connectivity.weights,
                    next_layer_states, hidden_pre_update_weights,
                )

            # Propagate signals backward
            updated_backward = jax.vmap(
                user_fns.backward_signal_fn, in_axes=(0, 0, None)
            )(layer_states, layer_indices, next_layer_states)

            layer_states = eqx.tree_at(
                lambda s: s.backward_state, layer_states, updated_backward)

            # Save pre-update weights for this layer
            hidden_pre_update_weights = layer_states.connectivity.weights

            # Run neuron update
            layer_states = jax.vmap(user_fns.neuron_update_fn)(layer_states)
            hidden_states = self._set_layer_states(hidden_states, start, end, layer_states)

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states)

    def _structure_update(
        self,
        structure_update_fn: Callable,
    ) -> Tuple['Network', list]:
        """Run structure update for each hidden layer (last to first)."""
        hidden_states = self.hidden_states
        output_states = self.output_states
        structure_state = self.structure_state
        generation_specs = []

        for layer_k in range(self.max_layers - 1, -1, -1):
            start, end = self.layer_boundaries[layer_k]
            layer_states = self._get_layer_states(hidden_states, start, end)

            if layer_k == self.max_layers - 1:
                next_layer_states = output_states
            else:
                next_start, next_end = self.layer_boundaries[layer_k + 1]
                next_layer_states = self._get_layer_states(hidden_states, next_start, next_end)

            structure_state, prune_mask, n_generate = structure_update_fn(
                layer_states, next_layer_states, structure_state)

            hidden_states, output_states = self._apply_pruning(
                hidden_states, output_states, self.n_inputs, start, end, prune_mask)

            generation_specs.append((layer_k, n_generate))

        updated = tree_replace(self, hidden_states=hidden_states, output_states=output_states,
                               structure_state=structure_state)
        return updated, generation_specs

    def _generate_neurons(
        self,
        generation_specs: list,
        init_neuron_fn: Callable,
        key: PRNGKeyArray,
    ) -> 'Network':
        """Generate new neurons as specified by the structure update."""
        hidden_states = self.hidden_states
        output_states = self.output_states

        for layer_k, n_generate in generation_specs:
            start, end = self.layer_boundaries[layer_k]
            key, gen_key = jax.random.split(key)

            if layer_k == 0:
                prior_ids = jnp.arange(self.n_inputs)
                prior_active = jnp.ones(self.n_inputs, dtype=bool)
            else:
                prev_start, prev_end = self.layer_boundaries[layer_k - 1]
                prior_ids = jnp.arange(prev_start, prev_end) + self.n_inputs
                prior_active = hidden_states.active_mask[prev_start:prev_end]

            for gen_idx in range(self.max_generate_per_step):
                key, unit_key = jax.random.split(key)

                layer_active = hidden_states.active_mask[start:end]
                slot = jnp.argmin(layer_active)
                has_slot = ~layer_active[slot]
                should_generate = has_slot & (gen_idx < n_generate)

                incoming_ids = jnp.zeros(self.max_connections, dtype=jnp.int32)
                conn_mask = jnp.zeros(self.max_connections, dtype=bool)
                weights = jnp.zeros(self.max_connections)

                n_prior = prior_ids.shape[0]
                n_to_connect = jnp.minimum(n_prior, self.max_connections)
                for c in range(self.max_connections):
                    is_valid = (c < n_to_connect) & (c < n_prior) & prior_active[jnp.minimum(c, n_prior - 1)]
                    incoming_ids = incoming_ids.at[c].set(
                        jnp.where(is_valid, prior_ids[jnp.minimum(c, n_prior - 1)], 0))
                    conn_mask = conn_mask.at[c].set(is_valid)

                connectivity = tree_replace(
                    ConnectivityState(self.max_connections),
                    incoming_ids=incoming_ids, weights=weights,
                    active_connection_mask=conn_mask,
                )

                abs_index = start + slot + self.n_inputs
                new_neuron = init_neuron_fn(connectivity, abs_index, unit_key)

                new_neuron_active = tree_replace(new_neuron, active_mask=jnp.array(True))
                layer_states = self._get_layer_states(hidden_states, start, end)

                updated_layer = jax.tree.map(
                    lambda full, single: jnp.where(should_generate, full.at[slot].set(single), full),
                    layer_states, new_neuron_active,
                )
                hidden_states = self._set_layer_states(hidden_states, start, end, updated_layer)

                if self.auto_connect_to_output:
                    output_states = self._auto_connect_to_output(
                        output_states, abs_index, should_generate)

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states)

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _add_connection(
        neuron_states: NeuronState,
        to_idx: Int[Array, ''],
        from_idx: Int[Array, ''],
        weight: Float[Array, ''],
    ) -> Tuple[NeuronState, Bool[Array, '']]:
        """Add a connection to a neuron in the given state array.

        Returns the updated states and a boolean indicating success.
        """
        connectivity = neuron_states.connectivity
        conn_mask = connectivity.active_connection_mask[to_idx]
        slot = jnp.argmin(conn_mask)
        has_slot = ~conn_mask[slot]

        updated = tree_replace(
            connectivity,
            incoming_ids=connectivity.incoming_ids.at[to_idx, slot].set(from_idx),
            weights=connectivity.weights.at[to_idx, slot].set(weight),
            active_connection_mask=connectivity.active_connection_mask.at[to_idx, slot].set(True),
        )
        new_connectivity = jax.tree.map(
            lambda u, o: jnp.where(has_slot, u, o),
            updated, connectivity,
        )
        return eqx.tree_at(lambda s: s.connectivity, neuron_states, new_connectivity), has_slot

    @staticmethod
    def _remove_connection(
        neuron_states: NeuronState,
        neuron_idx: Int[Array, ''],
        connection_slot: int,
    ) -> NeuronState:
        """Deactivate a connection at the given slot of a neuron."""
        return eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, neuron_states,
            neuron_states.connectivity.active_connection_mask.at[
                neuron_idx, connection_slot].set(False),
        )

    @staticmethod
    def _get_layer_states(neuron_states: NeuronState, start: int, end: int) -> NeuronState:
        return jax.tree.map(lambda x: x[start:end], neuron_states)

    @staticmethod
    def _set_layer_states(
        neuron_states: NeuronState, start: int, end: int, layer_states: NeuronState,
    ) -> NeuronState:
        return jax.tree.map(
            lambda full, layer: full.at[start:end].set(layer),
            neuron_states, layer_states,
        )

    @staticmethod
    def _apply_pruning(
        hidden_states: NeuronState,
        output_states: NeuronState,
        n_inputs: int,
        start: int,
        end: int,
        prune_mask: Bool[Array, 'layer_size'],
    ) -> Tuple[NeuronState, NeuronState]:
        """Apply a pruning mask to a hidden layer."""
        # Deactivate pruned neurons
        current_active = hidden_states.active_mask[start:end]
        new_active = current_active & ~prune_mask
        hidden_states = eqx.tree_at(
            lambda s: s.active_mask, hidden_states,
            hidden_states.active_mask.at[start:end].set(new_active),
        )

        # Deactivate connections in other neurons that point to pruned neurons
        abs_indices = jnp.arange(start, end) + n_inputs
        pruned_abs = jnp.where(prune_mask, abs_indices, -1)

        hidden_incoming = hidden_states.connectivity.incoming_ids
        hidden_conn_mask = hidden_states.connectivity.active_connection_mask
        is_pruned_hidden = jnp.isin(hidden_incoming, pruned_abs)
        hidden_conn_mask = jnp.where(is_pruned_hidden, False, hidden_conn_mask)
        hidden_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, hidden_states, hidden_conn_mask)

        output_incoming = output_states.connectivity.incoming_ids
        output_conn_mask = output_states.connectivity.active_connection_mask
        is_pruned_output = jnp.isin(output_incoming, pruned_abs)
        output_conn_mask = jnp.where(is_pruned_output, False, output_conn_mask)
        output_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, output_states, output_conn_mask)

        return hidden_states, output_states

    @staticmethod
    def _auto_connect_to_output(
        output_states: NeuronState,
        neuron_abs_index: Int[Array, ''],
        should_connect: Bool[Array, ''],
    ) -> NeuronState:
        """Add a connection from a new hidden neuron to each output unit."""
        n_outputs = output_states.active_mask.shape[0]
        for out_idx in range(n_outputs):
            conn_mask = output_states.connectivity.active_connection_mask[out_idx]
            slot = jnp.argmin(conn_mask)
            has_slot = ~conn_mask[slot]
            do_connect = should_connect & has_slot

            output_states = eqx.tree_at(
                lambda s: s.connectivity.incoming_ids, output_states,
                jnp.where(do_connect,
                          output_states.connectivity.incoming_ids.at[out_idx, slot].set(neuron_abs_index),
                          output_states.connectivity.incoming_ids),
            )
            output_states = eqx.tree_at(
                lambda s: s.connectivity.active_connection_mask, output_states,
                jnp.where(do_connect,
                          output_states.connectivity.active_connection_mask.at[out_idx, slot].set(True),
                          output_states.connectivity.active_connection_mask),
            )

        return output_states
