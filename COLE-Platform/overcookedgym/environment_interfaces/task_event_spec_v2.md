# Task Event Specification v2

## Scope

This specification defines successful environment events. Events describe
verified state transitions, not player intent. The detector never uses an
action value to infer success.

## Common event format

```json
{
  "event_type": "ingredient_acquired",
  "timestep": 0,
  "agent_index": 0,
  "details": {}
}
```

- `timestep` is the transition index within one episode and starts at zero.
- `agent_index` is the real Overcooked player index, not ego/alt ordering.
- `details` contains the state evidence specific to the event.

## Event rules

### ingredient_acquired

Required evidence:

- The agent held no object before the transition.
- The agent holds an `onion` or `tomato` after the transition.

When terrain and object state prove it, `details.source` is `counter` or
`dispenser`. Tomato is supported because the repository includes tomato
layouts, although the initial handoff examples mention onion only.

### ingredient_put_in_pot

Required evidence:

- Before the transition the agent holds an onion or tomato.
- After the transition that agent holds no object.
- The pot the agent faces changes from empty to one matching ingredient, or
  its matching soup ingredient count increases by exactly one.

### plate_acquired

Required evidence:

- The agent held no object before the transition.
- The agent holds a `dish` after the transition.

### soup_plated

Required evidence:

- The agent holds a dish before and soup after the transition.
- The ready soup in the faced pot disappears.
- Soup type, item count, and cook time match before-pot and after-held state.
- Item count is complete and cook time is ready.

### soup_delivered

Required evidence:

- The agent holds complete, cooked soup before and no object after.
- The agent faces a serving location.
- `team_reward` is positive.
- For a finite order list, the first order matches the soup (or is `any`) and
  the after-order list equals the remaining suffix.
- For an unlimited `order_list=null`, positive reward is mandatory because
  state alone cannot prove that the disappearance was a rewarded delivery.

## Non-events and unreliable cases

- `interact` without the required state delta is not an event.
- Movement toward a dispenser, pot, dish, soup, or serving location is not an
  event.
- A held ingredient disappearing without a matching pot delta is not
  `ingredient_put_in_pot`.
- Dish-to-soup without a matching ready pot disappearance is not
  `soup_plated`.
- Soup disappearance at serving without positive reward evidence is not
  emitted as `soup_delivered`; it is intentionally left unclassified.

## Trajectory evidence

- `step_records[n].state` is the state before transition `n`.
- The next pre-state is `step_records[n+1].state` when it exists.
- `ep_final_states[episode]` stores the state after the final transition, so
  the final event remains independently auditable.
- Events are duplicated in `ep_events` for compatibility and must exactly
  match the corresponding step record.
