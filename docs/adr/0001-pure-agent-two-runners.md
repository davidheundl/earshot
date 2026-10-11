# The Agent is pure and tick-driven; two runners drive it

`Agent.step(t, sensed, inbox) -> (command, outbox, telemetry)` holds the whole brain and touches no clock, socket, thread or Webots API. A lockstep runner advances world, radio and all six Agents in one process on one clock; a live runner puts each Agent in its own process behind ZeroMQ. We chose this because the 100-run evaluation must be reproducible (same seed, same result), and ZeroMQ delivery order between processes is not, while the demo still needs separate processes to make "every drone decides locally" visible.

## Consequences

Any logic that sneaks real time or I/O into `earshot/agent/` breaks the evaluation's determinism; `tests/test_lockstep.py` guards this by running the same seed twice.
