from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from plastax.states import (
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
    connectivity_init_fn: Callable = eqx.field(static=True)
    state_init_fn: Callable = eqx.field(static=True)
    compute_output_error_fn: Callable = eqx.field(static=True)
    output_forward_fn: Callable | None = eqx.field(static=True, default=None)
    output_backward_signal_fn: Callable | None = eqx.field(static=True, default=None)
    output_neuron_update_fn: Callable | None = eqx.field(static=True, default=None)
    output_state_init_fn: Callable | None = eqx.field(static=True, default=None)


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
    hidden_neuron_cls: type = eqx.field(static=True)
    output_neuron_cls: type = eqx.field(static=True)

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
        hidden_neuron_cls: type[NeuronState],
        state_update_fns: StateUpdateFunctions,
        structure_state: StructureUpdateState,
        output_neuron_cls: type[NeuronState] = None,
        max_generate_per_step: int = 0,
        auto_connect_to_output: bool = False,
    ):
        """Create a Network with all hidden neurons inactive.

        Uses the given NeuronState classes (or subclasses) to construct the
        initial state arrays.  max_connections and max_output_connections are
        derived from the constructed neurons' weight shapes.  Output neurons
        are activated.  If auto_connect_to_output is True, outputs are wired
        to receive from all input units (with zero weights).
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_hidden_per_layer = max_hidden_per_layer
        self.max_layers = max_layers
        self.max_generate_per_step = max_generate_per_step
        self.auto_connect_to_output = auto_connect_to_output
        self.state_update_fns = state_update_fns

        if output_neuron_cls is None:
            output_neuron_cls = hidden_neuron_cls
        self.hidden_neuron_cls = hidden_neuron_cls
        self.output_neuron_cls = output_neuron_cls

        total_hidden = max_hidden_per_layer * max_layers
        self.total_hidden = total_hidden
        self.total_neurons = n_inputs + total_hidden + n_outputs
        self.layer_boundaries = tuple(
            (k * max_hidden_per_layer, (k + 1) * max_hidden_per_layer)
            for k in range(max_layers)
        )

        # Input activations default to zeros
        self.input_values = jnp.zeros(n_inputs)

        # Create initial states by vmapping over constructors
        hidden_states = jax.vmap(lambda _: hidden_neuron_cls())(jnp.arange(total_hidden))
        output_states = jax.vmap(lambda _: output_neuron_cls())(jnp.arange(n_outputs))
        self.max_connections = hidden_states.weights.shape[-1]
        self.max_output_connections = output_states.weights.shape[-1]

        # Activate output neurons
        output_states = eqx.tree_at(
            lambda s: s.active_mask, output_states,
            jnp.ones(n_outputs, dtype=bool),
        )

        # Optionally connect output neurons to input neurons
        if auto_connect_to_output:
            input_ids = jnp.arange(n_inputs, dtype=jnp.int32)
            output_states = tree_replace(
                output_states,
                incoming_ids=output_states.incoming_ids.at[:, :n_inputs].set(input_ids),
                weights=output_states.weights.at[:, :n_inputs].set(0.0),
                active_connection_mask=output_states.active_connection_mask.at[:, :n_inputs].set(True),
            )

        self.hidden_states = hidden_states
        self.output_states = output_states
        self.structure_state = structure_state

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def _get_output_state_init_fn(self) -> Callable:
        """Return output_state_init_fn, falling back to state_init_fn."""
        fns = self.state_update_fns
        return fns.output_state_init_fn if fns.output_state_init_fn is not None else fns.state_init_fn

    def step(
        self,
        inputs: Float[Array, 'n_inputs'],
        targets: Float[Array, 'n_outputs'],
        key: PRNGKeyArray,
    ) -> 'Network':
        """Run one full step: forward -> backward -> structure update -> generation."""
        network = tree_replace(self, input_values=inputs)
        network = network._forward_pass()
        network = network._backward_pass(targets)
        network = network._structure_update(key)
        return network

    def get_units_in_layer(self, layer_k: int) -> Int[Array, 'max_hidden_per_layer']:
        """Return absolute indices for all neuron slots in hidden layer layer_k.

        Supports negative indexing: -1 is the last layer.
        """
        if layer_k < 0:
            layer_k = self.max_layers + layer_k
        start, end = self.layer_boundaries[layer_k]
        return jnp.arange(start, end) + self.n_inputs

    def add_layer(
        self,
        n_units: int,
        key: PRNGKeyArray,
        from_indices: Int[Array, 'n_from'] | None = None,
        connect_to_output: bool = False,
    ) -> 'Network':
        """Add a layer of neurons using connectivity_init_fn + state_init_fn.

        Determines per-unit connectivity from the given pool of source neurons
        via connectivity_init_fn, then initializes state via state_init_fn.
        The target layer is inferred from the source neurons.

        Not JIT-compatible (uses Python-level control flow for layer inference).
        Should be called sequentially from lower to higher layers so that each
        layer's active neurons are visible to subsequent layers.

        Args:
            n_units: Number of neurons to create.
            key: PRNG key.
            from_indices: Pool of absolute neuron indices to connect from.
                If None, uses all active units from the latest active hidden
                layer, or input indices if no hidden layers are active.
            connect_to_output: If True, connect all new neurons to outputs.
        """
        fns = self.state_update_fns

        # Determine source pool
        if from_indices is None:
            from_indices = self._get_latest_active_indices()

        # Build connectable mask restricted to from_indices
        connectable_mask = jnp.zeros(self.n_inputs + self.total_hidden, dtype=bool)
        connectable_mask = connectable_mask.at[from_indices].set(True)

        # Infer target layer from max source index
        max_from_id = int(jnp.max(from_indices))
        target_layer = 0 if max_from_id < self.n_inputs else (
            (max_from_id - self.n_inputs) // self.max_hidden_per_layer + 1)

        start, end = self.layer_boundaries[target_layer]
        abs_indices = jnp.arange(start, start + n_units) + self.n_inputs

        # Split keys
        keys = jax.random.split(key, 2 * n_units + 1)
        conn_keys = keys[:n_units]
        state_keys = keys[n_units:2 * n_units]
        auto_connect_key = keys[2 * n_units]

        # Determine connectivity
        incoming_ids, active_conn_masks = jax.vmap(
            fns.connectivity_init_fn, in_axes=(None, 0, None, 0)
        )(connectable_mask, abs_indices, self.max_connections, conn_keys)

        # Create blank neurons, set connectivity, init state
        blanks = jax.vmap(lambda _: self.hidden_neuron_cls())(jnp.arange(n_units))
        blanks = tree_replace(
            blanks,
            active_mask=jnp.ones(n_units, dtype=bool),
            incoming_ids=incoming_ids,
            active_connection_mask=active_conn_masks,
        )
        neuron_states = jax.vmap(fns.state_init_fn)(blanks, state_keys)

        # Place into target layer slots
        layer_active = self.hidden_states.active_mask[start:end]
        sorted_slots = jnp.argsort(layer_active.astype(jnp.int32))[:n_units]

        hidden_states = jax.tree.map(
            lambda full, vals: full.at[start + sorted_slots].set(vals),
            self.hidden_states, neuron_states,
        )

        output_states = self.output_states
        if connect_to_output:
            abs_placed = start + sorted_slots + self.n_inputs
            has_slot = jnp.ones(n_units, dtype=bool)
            output_states = self._auto_connect_to_output(
                output_states, abs_placed, has_slot, auto_connect_key)

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states)

    def _get_latest_active_indices(self) -> Int[Array, 'n']:
        """Find the latest hidden layer with active neurons and return their indices.

        If no hidden layers are active, returns input indices. Not JIT-compatible.
        """
        for layer_k in range(self.max_layers - 1, -1, -1):
            start, end = self.layer_boundaries[layer_k]
            layer_active = self.hidden_states.active_mask[start:end]
            if bool(jnp.any(layer_active)):
                all_indices = jnp.arange(start, end) + self.n_inputs
                return all_indices[layer_active]
        return jnp.arange(self.n_inputs, dtype=jnp.int32)

    def connect_to_output(
        self,
        from_indices: Int[Array, 'n_from'],
        key: PRNGKeyArray,
    ) -> 'Network':
        """Connect hidden neurons to all output neurons.

        For each output neuron, finds inactive connection slots and assigns one
        per from_index. Weights for new connections are initialized via
        output_state_init_fn (or state_init_fn as fallback).

        Args:
            from_indices: Absolute indices of neurons to connect from.
            key: PRNG key for weight initialization.
        """
        n_from = from_indices.shape[0]
        output_states = self.output_states

        # For each output neuron, find n_from inactive slots via argsort
        sorted_indices = jnp.argsort(
            output_states.active_connection_mask.astype(jnp.int32), axis=1)
        slots = sorted_indices[:, :n_from]

        out_idx = jnp.arange(self.n_outputs)[:, None]
        has_slot = ~output_states.active_connection_mask[out_idx, slots]

        # Set incoming IDs and connection mask
        new_ids = jnp.where(has_slot, from_indices[None, :],
                            output_states.incoming_ids[out_idx, slots])
        new_mask = jnp.where(has_slot, True,
                             output_states.active_connection_mask[out_idx, slots])

        output_states = tree_replace(
            output_states,
            incoming_ids=output_states.incoming_ids.at[out_idx, slots].set(new_ids),
            active_connection_mask=output_states.active_connection_mask.at[out_idx, slots].set(new_mask),
        )

        # Initialize weights via state_init_fn
        init_fn = self._get_output_state_init_fn()
        keys = jax.random.split(key, self.n_outputs)
        initialized = jax.vmap(init_fn)(output_states, keys)

        # Extract only new slots' weights, preserving existing
        init_weights = initialized.weights[out_idx, slots]
        existing_weights = self.output_states.weights[out_idx, slots]
        new_weights = jnp.where(has_slot, init_weights, existing_weights)
        output_states = tree_replace(
            output_states,
            weights=output_states.weights.at[out_idx, slots].set(new_weights),
        )

        return tree_replace(self, output_states=output_states)

    # -----------------------------------------------------------------------
    # Ops (public)
    # -----------------------------------------------------------------------

    def add_unit(
        self,
        incoming_ids: Int[Array, 'max_connections'],
        key: PRNGKeyArray,
        connect_to_output: bool = False,
    ) -> Tuple['Network', Bool[Array, '']]:
        """Insert a neuron into the correct hidden layer based on its incoming connections.

        The target layer is determined automatically: if the neuron's active
        incoming connections reference layer *k* (or the inputs), the neuron is
        placed in layer *k+1* (or layer 0).

        Args:
            incoming_ids: Absolute indices of source neurons, padded with -1
                for inactive connection slots.
            key: PRNG key for state initialization.
            connect_to_output: If True, also connect the new neuron to all outputs.

        Returns the updated network and a boolean indicating success (False if
        the target layer had no inactive slots).
        """
        state_key, auto_connect_key = jax.random.split(key)

        # Derive connectivity
        active_connection_mask = incoming_ids >= 0
        safe_ids = jnp.where(active_connection_mask, incoming_ids, 0)

        # Create blank neuron, set connectivity, init state
        blank = self.hidden_neuron_cls()
        neuron_state = tree_replace(
            blank,
            active_mask=jnp.array(True),
            incoming_ids=safe_ids,
            active_connection_mask=active_connection_mask,
        )
        neuron_state = self.state_update_fns.state_init_fn(neuron_state, state_key)

        # Determine target layer from max active incoming ID
        active_ids = jnp.where(active_connection_mask, incoming_ids, -1)
        max_id = jnp.max(active_ids)

        layer_idx = jnp.where(
            max_id < self.n_inputs, 0,
            (max_id - self.n_inputs) // self.max_hidden_per_layer + 1,
        )
        start = layer_idx * self.max_hidden_per_layer

        # Find first inactive slot in the target layer
        layer_active = jax.lax.dynamic_slice(
            self.hidden_states.active_mask, (start,), (self.max_hidden_per_layer,))
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
            output_states = self._auto_connect_to_output(
                output_states, abs_index[None], has_slot[None], auto_connect_key)

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states), has_slot

    def add_units(
        self,
        incoming_ids_batch: Int[Array, 'n_units max_connections'],
        key: PRNGKeyArray,
        connect_to_output: bool = False,
    ) -> Tuple['Network', Bool[Array, 'n_units']]:
        """Insert multiple neurons into the correct hidden layer.

        All neurons must target the same layer (inferred from the max incoming
        ID across the batch).

        Args:
            incoming_ids_batch: (n_units, max_connections) array of source
                neuron absolute indices, padded with -1 for inactive slots.
            key: PRNG key.
            connect_to_output: If True, connect all new neurons to outputs.

        Returns (updated_network, success_mask).
        """
        n_units = incoming_ids_batch.shape[0]
        active_masks = incoming_ids_batch >= 0
        safe_ids = jnp.where(active_masks, incoming_ids_batch, 0)

        # Split keys
        keys = jax.random.split(key, n_units + 1)
        state_keys = keys[:n_units]
        auto_connect_key = keys[n_units]

        # Create blank neurons, set connectivity, init state
        blanks = jax.vmap(lambda _: self.hidden_neuron_cls())(jnp.arange(n_units))
        neuron_states = tree_replace(
            blanks,
            active_mask=jnp.ones(n_units, dtype=bool),
            incoming_ids=safe_ids,
            active_connection_mask=active_masks,
        )
        neuron_states = jax.vmap(self.state_update_fns.state_init_fn)(neuron_states, state_keys)

        # Infer target layer from global max incoming ID
        all_active_ids = jnp.where(active_masks, incoming_ids_batch, -1)
        max_id = jnp.max(all_active_ids)
        layer_idx = jnp.where(
            max_id < self.n_inputs, 0,
            (max_id - self.n_inputs) // self.max_hidden_per_layer + 1,
        )
        start = layer_idx * self.max_hidden_per_layer

        # Find inactive slots in the target layer
        layer_active = jax.lax.dynamic_slice(
            self.hidden_states.active_mask, (start,), (self.max_hidden_per_layer,))
        sorted_slots = jnp.argsort(layer_active.astype(jnp.int32))[:n_units]
        has_slot = ~layer_active[sorted_slots]
        hidden_rel = start + sorted_slots

        # Scatter neurons into slots
        hidden_states = jax.tree.map(
            lambda full, vals: full.at[hidden_rel].set(
                jnp.where(
                    has_slot.reshape((-1,) + (1,) * (vals.ndim - 1)),
                    vals, full[hidden_rel],
                )
            ),
            self.hidden_states, neuron_states,
        )

        output_states = self.output_states
        if connect_to_output:
            abs_indices = hidden_rel + self.n_inputs
            output_states = self._auto_connect_to_output(
                output_states, abs_indices, has_slot, auto_connect_key)

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
        hidden_match = hidden_states.incoming_ids == neuron_abs_idx
        hidden_states = eqx.tree_at(
            lambda s: s.active_connection_mask, hidden_states,
            jnp.where(hidden_match, False, hidden_states.active_connection_mask),
        )

        # Deactivate connections pointing to this neuron in output layer
        output_match = output_states.incoming_ids == neuron_abs_idx
        output_states = eqx.tree_at(
            lambda s: s.active_connection_mask, output_states,
            jnp.where(output_match, False, output_states.active_connection_mask),
        )

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states)

    def add_connection_to_hidden(
        self,
        from_idx: Int[Array, ''],
        to_idx: Int[Array, ''],
        key: PRNGKeyArray,
    ) -> Tuple['Network', Bool[Array, '']]:
        """Add a connection to a hidden neuron, initializing weight via state_init_fn.

        Both indices are relative to the hidden state array. Returns the
        updated network and a boolean indicating success (False if the neuron
        had no inactive connection slots).
        """
        hidden_states, success, slot = self._add_connection(
            self.hidden_states, to_idx, from_idx)

        # Initialize weight for the new connection
        neuron = jax.tree.map(lambda x: x[to_idx], hidden_states)
        initialized = self.state_update_fns.state_init_fn(neuron, key)
        new_weight = initialized.weights[slot]

        hidden_states = tree_replace(
            hidden_states,
            weights=jnp.where(
                success,
                hidden_states.weights.at[to_idx, slot].set(new_weight),
                hidden_states.weights,
            ),
        )
        return tree_replace(self, hidden_states=hidden_states), success

    def add_connection_to_output(
        self,
        from_idx: Int[Array, ''],
        to_idx: Int[Array, ''],
        key: PRNGKeyArray,
    ) -> Tuple['Network', Bool[Array, '']]:
        """Add a connection to an output neuron, initializing weight via output_state_init_fn.

        Both indices are relative to the output state array. Returns the
        updated network and a boolean indicating success (False if the neuron
        had no inactive connection slots).
        """
        output_states, success, slot = self._add_connection(
            self.output_states, to_idx, from_idx)

        # Initialize weight for the new connection
        init_fn = self._get_output_state_init_fn()
        neuron = jax.tree.map(lambda x: x[to_idx], output_states)
        initialized = init_fn(neuron, key)
        new_weight = initialized.weights[slot]

        output_states = tree_replace(
            output_states,
            weights=jnp.where(
                success,
                output_states.weights.at[to_idx, slot].set(new_weight),
                output_states.weights,
            ),
        )
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
            self.hidden_states.activation_value,
            self.output_states.activation_value,
        ])

    def _forward_pass(self) -> 'Network':
        """Run forward pass layer by layer (global mode)."""
        fns = self.state_update_fns
        hidden_states = self.hidden_states
        all_activations = self._build_all_activations()

        for layer_k in range(self.max_layers):
            start, end = self.layer_boundaries[layer_k]
            layer_states = self._get_layer_states(hidden_states, start, end)

            incoming_acts = all_activations[layer_states.incoming_ids]
            new_activations, updated_states = jax.vmap(fns.forward_fn)(layer_states, incoming_acts)
            new_activations = new_activations * layer_states.active_mask

            hidden_states = self._set_layer_states(hidden_states, start, end, updated_states)
            abs_start = self.n_inputs + start
            abs_end = self.n_inputs + end
            all_activations = all_activations.at[abs_start:abs_end].set(new_activations)

        output_incoming_acts = all_activations[self.output_states.incoming_ids]
        out_fn = fns.output_forward_fn if fns.output_forward_fn is not None else fns.forward_fn
        _, updated_output_states = jax.vmap(out_fn)(self.output_states, output_incoming_acts)

        return tree_replace(self, hidden_states=hidden_states, output_states=updated_output_states)

    def _backward_pass(self, targets: Float[Array, 'n_outputs']) -> 'Network':
        """Run backward pass: output error -> backward signal -> neuron update, layer by layer."""
        fns = self.state_update_fns
        hidden_states = self.hidden_states
        output_states = self.output_states

        # 1. Compute output error
        output_activations = output_states.activation_value
        output_error = fns.compute_output_error_fn(output_activations, targets)
        output_states = eqx.tree_at(
            lambda s: s.error_signal,
            output_states,
            output_error,
        )

        # 2. Top-down: propagate error (with activation derivative) then update weights
        out_update_fn = (fns.output_neuron_update_fn
                         if fns.output_neuron_update_fn is not None
                         else fns.neuron_update_fn)
        out_backward_fn = (fns.output_backward_signal_fn
                           if fns.output_backward_signal_fn is not None
                           else fns.backward_signal_fn)

        for layer_k in range(self.max_layers, -1, -1):
            # Get this layer's states
            if layer_k == self.max_layers:
                current_states = output_states
                back_fn = out_backward_fn
            else:
                start, end = self.layer_boundaries[layer_k]
                current_states = self._get_layer_states(hidden_states, start, end)
                back_fn = fns.backward_signal_fn

            # Propagate error to the layer below (using original weights)
            if layer_k > 0:
                below_start, below_end = self.layer_boundaries[layer_k - 1]
                below_states = self._get_layer_states(hidden_states, below_start, below_end)
                below_indices = jnp.arange(below_start, below_end) + self.n_inputs
                updated_errors = jax.vmap(
                    back_fn, in_axes=(0, 0, None)
                )(below_states, below_indices, current_states)
                below_states = eqx.tree_at(
                    lambda s: s.error_signal, below_states, updated_errors)
                hidden_states = self._set_layer_states(
                    hidden_states, below_start, below_end, below_states)

            # Update this layer's weights
            if layer_k == self.max_layers:
                output_states = jax.vmap(out_update_fn)(output_states)
            else:
                current_states = jax.vmap(fns.neuron_update_fn)(current_states)
                hidden_states = self._set_layer_states(hidden_states, start, end, current_states)

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states)

    def _build_connectable_mask(
        self,
        hidden_states: NeuronState,
        layer_k: int,
    ) -> Bool[Array, 'n_inputs+total_hidden']:
        """Mask over global index space: inputs (always True) + active hidden neurons
        in all layers prior to layer_k."""
        prior_end = self.layer_boundaries[layer_k - 1][1] if layer_k > 0 else 0
        return jnp.concatenate([
            jnp.ones(self.n_inputs, dtype=bool), # inputs are always connectable
            hidden_states.active_mask[:prior_end], # all active neurons in layers prior to layer_k
            jnp.zeros(self.total_hidden - prior_end, dtype=bool), # no neurons in layer_k and beyond
        ])

    def _generate_into_layer(
        self,
        hidden_states: NeuronState,
        start: int,
        end: int,
        n_generate,
        connectable_mask: Bool[Array, 'n_inputs+total_hidden'],
        unit_keys: PRNGKeyArray,
    ) -> Tuple[NeuronState, Int[Array, 'n_gen'], Bool[Array, 'n_gen']]:
        """Generate new neurons into inactive slots via connectivity_init_fn + state_init_fn.

        Returns updated hidden_states, absolute indices of the slots used,
        and a mask of which slots actually had neurons generated.
        """
        fns = self.state_update_fns
        n_gen = self.max_generate_per_step
        layer_active = hidden_states.active_mask[start:end]
        slots = jnp.argsort(layer_active.astype(jnp.int32))[:n_gen]
        should_generate = ~layer_active[slots] & (jnp.arange(n_gen) < n_generate)

        abs_indices = start + slots + self.n_inputs

        # Split keys: one for connectivity, one for state init
        split_keys = jax.vmap(jax.random.split)(unit_keys)
        conn_keys = split_keys[:, 0]
        state_keys = split_keys[:, 1]

        # Step 1: Determine connectivity
        incoming_ids, active_conn_masks = jax.vmap(
            fns.connectivity_init_fn, in_axes=(None, 0, None, 0)
        )(connectable_mask, abs_indices, self.max_connections, conn_keys)

        # Step 2: Create blank neurons with connectivity set
        blanks = jax.vmap(lambda _: self.hidden_neuron_cls())(jnp.arange(n_gen))
        blanks = tree_replace(
            blanks,
            active_mask=jnp.ones(n_gen, dtype=bool),
            incoming_ids=incoming_ids,
            active_connection_mask=active_conn_masks,
        )

        # Step 3: Initialize state (weights, etc.)
        new_neurons = jax.vmap(fns.state_init_fn)(blanks, state_keys)

        # Conditionally scatter into the layer
        layer = self._get_layer_states(hidden_states, start, end)
        existing = jax.tree.map(lambda x: x[slots], layer)
        masked = jax.tree.map(
            lambda e, n: jnp.where(
                should_generate.reshape((-1,) + (1,) * (e.ndim - 1)), n, e
            ),
            existing, new_neurons,
        )
        layer = jax.tree.map(lambda full, vals: full.at[slots].set(vals), layer, masked)
        hidden_states = self._set_layer_states(hidden_states, start, end, layer)
        return hidden_states, abs_indices, should_generate

    def _structure_update(self, key: PRNGKeyArray) -> 'Network':
        """Run structure update for each hidden layer (last to first).

        For each layer: decide what to prune/generate, apply pruning,
        then immediately generate new neurons so that lower layers can
        see the updated state from upper layers.
        """
        fns = self.state_update_fns
        hidden_states = self.hidden_states
        output_states = self.output_states
        structure_state = self.structure_state

        for layer_k in range(self.max_layers - 1, -1, -1):
            start, end = self.layer_boundaries[layer_k]
            layer_states = self._get_layer_states(hidden_states, start, end)

            # Get the layer above (output or next hidden) for structure decisions
            if layer_k == self.max_layers - 1:
                next_layer_states = output_states
            else:
                next_start, next_end = self.layer_boundaries[layer_k + 1]
                next_layer_states = self._get_layer_states(hidden_states, next_start, next_end)

            # Decide which neurons to prune and how many to generate
            structure_state, prune_mask, n_generate = fns.structure_update_fn(
                layer_states, next_layer_states, structure_state)

            # Prune neurons and clean up dangling connections
            hidden_states, output_states = self._apply_pruning(
                hidden_states, output_states, self.n_inputs, start, end, prune_mask)

            # Generate new neurons for this layer
            n_gen = self.max_generate_per_step
            keys = jax.random.split(key, n_gen + 2)
            key = keys[0]
            auto_connect_key = keys[1]

            connectable_mask = self._build_connectable_mask(hidden_states, layer_k)
            hidden_states, abs_indices, did_generate = self._generate_into_layer(
                hidden_states, start, end, n_generate, connectable_mask, keys[2:])

            # Wire output neurons to the new neurons if auto-connecting
            if self.auto_connect_to_output:
                output_states = self._auto_connect_to_output(
                    output_states, abs_indices, did_generate, auto_connect_key)

        return tree_replace(self, hidden_states=hidden_states, output_states=output_states,
                            structure_state=structure_state)

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _add_connection(
        neuron_states: NeuronState,
        to_idx: Int[Array, ''],
        from_idx: Int[Array, ''],
    ) -> Tuple[NeuronState, Bool[Array, ''], Int[Array, '']]:
        """Add a connection to a neuron in the given state array.

        Only sets connectivity (incoming_ids, active_connection_mask).
        Weight initialization is the caller's responsibility via state_init_fn.

        Returns (updated_states, success, slot_index).
        """
        conn_mask = neuron_states.active_connection_mask[to_idx]
        slot = jnp.argmin(conn_mask)
        has_slot = ~conn_mask[slot]

        updated = tree_replace(
            neuron_states,
            incoming_ids=neuron_states.incoming_ids.at[to_idx, slot].set(from_idx),
            active_connection_mask=neuron_states.active_connection_mask.at[to_idx, slot].set(True),
        )
        result = jax.tree.map(
            lambda u, o: jnp.where(has_slot, u, o),
            updated, neuron_states,
        )
        return result, has_slot, slot

    @staticmethod
    def _remove_connection(
        neuron_states: NeuronState,
        neuron_idx: Int[Array, ''],
        connection_slot: int,
    ) -> NeuronState:
        """Deactivate a connection at the given slot of a neuron."""
        return eqx.tree_at(
            lambda s: s.active_connection_mask, neuron_states,
            neuron_states.active_connection_mask.at[
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

        hidden_incoming = hidden_states.incoming_ids
        hidden_conn_mask = hidden_states.active_connection_mask
        is_pruned_hidden = jnp.isin(hidden_incoming, pruned_abs)
        hidden_conn_mask = jnp.where(is_pruned_hidden, False, hidden_conn_mask)
        hidden_states = eqx.tree_at(
            lambda s: s.active_connection_mask, hidden_states, hidden_conn_mask)

        output_incoming = output_states.incoming_ids
        output_conn_mask = output_states.active_connection_mask
        is_pruned_output = jnp.isin(output_incoming, pruned_abs)
        output_conn_mask = jnp.where(is_pruned_output, False, output_conn_mask)
        output_states = eqx.tree_at(
            lambda s: s.active_connection_mask, output_states, output_conn_mask)

        return hidden_states, output_states

    def _auto_connect_to_output(
        self,
        output_states: NeuronState,
        neuron_abs_indices: Int[Array, 'n'],
        should_connect: Bool[Array, 'n'],
        key: PRNGKeyArray,
    ) -> NeuronState:
        """Add connections from new hidden neurons to each output unit.

        For each output neuron, finds n inactive slots and assigns one per
        new hidden neuron, masked by should_connect. Initializes weights for
        new connections via output_state_init_fn (or state_init_fn).
        """
        n = neuron_abs_indices.shape[0]

        # For each output neuron, find n inactive slots via argsort
        sorted_indices = jnp.argsort(
            output_states.active_connection_mask.astype(jnp.int32), axis=1)
        slots = sorted_indices[:, :n]

        # Check which slots are actually inactive
        out_idx = jnp.arange(output_states.active_connection_mask.shape[0])[:, None]
        has_slot = ~output_states.active_connection_mask[out_idx, slots]
        do_connect = has_slot & should_connect[None, :]

        # Build updated values, keeping originals where we shouldn't connect
        existing_ids = output_states.incoming_ids[out_idx, slots]
        new_ids = jnp.where(do_connect, neuron_abs_indices[None, :], existing_ids)
        existing_mask = output_states.active_connection_mask[out_idx, slots]
        new_mask = jnp.where(do_connect, True, existing_mask)

        # Set connectivity first
        output_states = tree_replace(
            output_states,
            incoming_ids=output_states.incoming_ids.at[out_idx, slots].set(new_ids),
            active_connection_mask=output_states.active_connection_mask.at[out_idx, slots].set(new_mask),
        )

        # Initialize weights for new connections via state_init_fn
        init_fn = self._get_output_state_init_fn()
        n_outputs = output_states.active_connection_mask.shape[0]
        keys = jax.random.split(key, n_outputs)
        initialized = jax.vmap(init_fn)(output_states, keys)

        # Only use weights at newly connected slots
        init_weights = initialized.weights[out_idx, slots]
        existing_weights = output_states.weights[out_idx, slots]
        new_weights = jnp.where(do_connect, init_weights, existing_weights)

        return tree_replace(
            output_states,
            weights=output_states.weights.at[out_idx, slots].set(new_weights),
        )
