from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from dynamic_net.states import (
    BackwardPassState,
    ConnectivityState,
    NeuronState,
    StructureUpdateState,
    tree_replace,
)


# ---------------------------------------------------------------------------
# User function container
# ---------------------------------------------------------------------------

class UserFunctions(eqx.Module):
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

    # Dynamic (state)
    input_activations: Float[Array, 'n_inputs']
    hidden_states: NeuronState
    output_states: NeuronState

    def __init__(
        self,
        *,
        n_inputs: int,
        n_outputs: int,
        max_hidden_per_layer: int,
        max_layers: int,
        max_connections: int,
        max_output_connections: int,
        max_generate_per_step: int = 0,
        auto_connect_to_output: bool = False,
        input_activations: Float[Array, 'n_inputs'],
        hidden_states: NeuronState,
        output_states: NeuronState,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_hidden_per_layer = max_hidden_per_layer
        self.max_layers = max_layers
        self.max_connections = max_connections
        self.max_output_connections = max_output_connections
        self.max_generate_per_step = max_generate_per_step
        self.auto_connect_to_output = auto_connect_to_output

        total_hidden = max_hidden_per_layer * max_layers
        self.total_hidden = total_hidden
        self.total_neurons = n_inputs + total_hidden + n_outputs
        self.layer_boundaries = tuple(
            (k * max_hidden_per_layer, (k + 1) * max_hidden_per_layer)
            for k in range(max_layers)
        )

        self.input_activations = input_activations
        self.hidden_states = hidden_states
        self.output_states = output_states

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        n_inputs: int,
        n_outputs: int,
        max_hidden_per_layer: int,
        max_layers: int,
        max_connections: int,
        max_output_connections: int,
        hidden_neuron_template: NeuronState,
        output_neuron_template: NeuronState,
        key: PRNGKeyArray,
        max_generate_per_step: int = 0,
        auto_connect_to_output: bool = False,
    ) -> 'Network':
        """Create a Network with inputs directly connected to outputs, all hidden inactive."""
        total_hidden = max_hidden_per_layer * max_layers

        hidden_states = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (total_hidden,) + x.shape).copy(),
            hidden_neuron_template,
        )
        output_states = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_outputs,) + x.shape).copy(),
            output_neuron_template,
        )

        # Activate output neurons
        output_states = eqx.tree_at(
            lambda s: s.active_mask, output_states,
            jnp.ones(n_outputs, dtype=bool),
        )

        # Connect output neurons to input neurons
        output_incoming_ids = jnp.zeros((n_outputs, max_output_connections), dtype=jnp.int32)
        output_conn_mask = jnp.zeros((n_outputs, max_output_connections), dtype=bool)
        input_ids = jnp.arange(n_inputs, dtype=jnp.int32)
        for i in range(n_outputs):
            output_incoming_ids = output_incoming_ids.at[i, :n_inputs].set(input_ids)
            output_conn_mask = output_conn_mask.at[i, :n_inputs].set(True)

        output_states = eqx.tree_at(
            lambda s: s.connectivity.incoming_ids, output_states, output_incoming_ids)
        output_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, output_states, output_conn_mask)

        return cls(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            max_hidden_per_layer=max_hidden_per_layer,
            max_layers=max_layers,
            max_connections=max_connections,
            max_output_connections=max_output_connections,
            max_generate_per_step=max_generate_per_step,
            auto_connect_to_output=auto_connect_to_output,
            input_activations=jnp.zeros(n_inputs),
            hidden_states=hidden_states,
            output_states=output_states,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def step(
        self,
        inputs: Float[Array, 'n_inputs'],
        targets: Float[Array, 'n_outputs'],
        structure_state: StructureUpdateState,
        user_fns: UserFunctions,
        key: PRNGKeyArray,
    ) -> Tuple['Network', StructureUpdateState]:
        """Run one full step: forward -> backward -> structure update -> generation."""
        network = tree_replace(self, input_activations=inputs)
        network = network._forward_pass(user_fns.forward_fn, user_fns.output_forward_fn)
        network = network._backward_pass(targets, user_fns)
        network, structure_state, generation_specs = network._structure_update(
            structure_state, user_fns.structure_update_fn)
        network = network._generate_neurons(generation_specs, user_fns.init_neuron_fn, key)
        return network, structure_state

    # -----------------------------------------------------------------------
    # Ops (public)
    # -----------------------------------------------------------------------

    def add_unit(self, layer_idx: int, neuron_state: NeuronState) -> 'Network':
        """Insert a neuron into the first inactive slot of the given hidden layer."""
        start, end = self.layer_boundaries[layer_idx]
        layer_active = self.hidden_states.active_mask[start:end]

        slot = jnp.argmin(layer_active)
        has_slot = ~layer_active[slot]

        layer_states = self._get_layer_states(self.hidden_states, start, end)
        updated_layer = jax.tree.map(
            lambda full, single: jnp.where(has_slot, full.at[slot].set(single), full),
            layer_states, neuron_state,
        )
        hidden_states = self._set_layer_states(self.hidden_states, start, end, updated_layer)
        return tree_replace(self, hidden_states=hidden_states)

    def remove_unit(self, neuron_abs_idx: Int[Array, '']) -> 'Network':
        """Deactivate a hidden neuron and remove all connections to/from it."""
        hidden_states = self.hidden_states
        output_states = self.output_states
        hidden_rel = neuron_abs_idx - self.n_inputs

        # Deactivate the neuron
        hidden_states = eqx.tree_at(
            lambda s: s.active_mask, hidden_states,
            hidden_states.active_mask.at[hidden_rel].set(False),
        )

        # Deactivate its incoming connections
        new_conn_mask = hidden_states.connectivity.active_connection_mask.at[hidden_rel].set(
            jnp.zeros(self.max_connections, dtype=bool))
        hidden_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, hidden_states, new_conn_mask)

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
        to_hidden_rel: Int[Array, ''],
        weight: Float[Array, ''] = jnp.array(0.0),
    ) -> 'Network':
        """Add a connection to a hidden neuron (specified by hidden-relative index)."""
        hidden_states = self.hidden_states
        conn_mask = hidden_states.connectivity.active_connection_mask[to_hidden_rel]
        slot = jnp.argmin(conn_mask)
        has_slot = ~conn_mask[slot]

        new_ids = jnp.where(
            has_slot,
            hidden_states.connectivity.incoming_ids.at[to_hidden_rel, slot].set(from_idx),
            hidden_states.connectivity.incoming_ids,
        )
        new_weights = jnp.where(
            has_slot,
            hidden_states.connectivity.weights.at[to_hidden_rel, slot].set(weight),
            hidden_states.connectivity.weights,
        )
        new_mask = jnp.where(
            has_slot,
            hidden_states.connectivity.active_connection_mask.at[to_hidden_rel, slot].set(True),
            hidden_states.connectivity.active_connection_mask,
        )

        hidden_states = eqx.tree_at(
            lambda s: (s.connectivity.incoming_ids, s.connectivity.weights,
                       s.connectivity.active_connection_mask),
            hidden_states, (new_ids, new_weights, new_mask),
        )
        return tree_replace(self, hidden_states=hidden_states)

    def add_connection_to_output(
        self,
        from_idx: Int[Array, ''],
        to_output_rel: Int[Array, ''],
        weight: Float[Array, ''] = jnp.array(0.0),
    ) -> 'Network':
        """Add a connection to an output neuron (specified by output-relative index)."""
        output_states = self.output_states
        conn_mask = output_states.connectivity.active_connection_mask[to_output_rel]
        slot = jnp.argmin(conn_mask)
        has_slot = ~conn_mask[slot]

        new_ids = jnp.where(
            has_slot,
            output_states.connectivity.incoming_ids.at[to_output_rel, slot].set(from_idx),
            output_states.connectivity.incoming_ids,
        )
        new_weights = jnp.where(
            has_slot,
            output_states.connectivity.weights.at[to_output_rel, slot].set(weight),
            output_states.connectivity.weights,
        )
        new_mask = jnp.where(
            has_slot,
            output_states.connectivity.active_connection_mask.at[to_output_rel, slot].set(True),
            output_states.connectivity.active_connection_mask,
        )

        output_states = eqx.tree_at(
            lambda s: (s.connectivity.incoming_ids, s.connectivity.weights,
                       s.connectivity.active_connection_mask),
            output_states, (new_ids, new_weights, new_mask),
        )
        return tree_replace(self, output_states=output_states)

    def remove_connection_from_hidden(
        self, hidden_rel: Int[Array, ''], connection_slot: int,
    ) -> 'Network':
        """Deactivate a connection at the given slot of a hidden neuron."""
        hidden_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, self.hidden_states,
            self.hidden_states.connectivity.active_connection_mask.at[
                hidden_rel, connection_slot].set(False),
        )
        return tree_replace(self, hidden_states=hidden_states)

    def remove_connection_from_output(
        self, output_rel: Int[Array, ''], connection_slot: int,
    ) -> 'Network':
        """Deactivate a connection at the given slot of an output neuron."""
        output_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, self.output_states,
            self.output_states.connectivity.active_connection_mask.at[
                output_rel, connection_slot].set(False),
        )
        return tree_replace(self, output_states=output_states)

    # -----------------------------------------------------------------------
    # Internal: forward / backward / structure / generation
    # -----------------------------------------------------------------------

    def _build_all_activations(self) -> Float[Array, 'total_neurons']:
        return jnp.concatenate([
            self.input_activations,
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
        self, targets: Float[Array, 'n_outputs'], user_fns: UserFunctions,
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
        structure_state: StructureUpdateState,
        structure_update_fn: Callable,
    ) -> Tuple['Network', StructureUpdateState, list]:
        """Run structure update for each hidden layer (last to first)."""
        hidden_states = self.hidden_states
        output_states = self.output_states
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

        updated = tree_replace(self, hidden_states=hidden_states, output_states=output_states)
        return updated, structure_state, generation_specs

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

                connectivity = ConnectivityState(
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

        # Deactivate pruned neurons' own incoming connections
        pruned_conn_mask = hidden_states.connectivity.active_connection_mask[start:end]
        pruned_conn_mask = jnp.where(prune_mask[:, None], False, pruned_conn_mask)
        hidden_states = eqx.tree_at(
            lambda s: s.connectivity.active_connection_mask, hidden_states,
            hidden_states.connectivity.active_connection_mask.at[start:end].set(pruned_conn_mask),
        )

        # Deactivate connections in other layers that point to pruned neurons
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
