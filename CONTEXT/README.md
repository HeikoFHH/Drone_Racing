# CONTEXT/ — documents written for LLM agents

Rule for this repo (set by the owner): **`CONTEXT/` holds documents meant to be read by LLM agents/subagents**
(architecture maps, invariants, conventions, contracts). **`docs/` holds human-readable descriptions** of what was
built and why (guides, plans, test reports, figures). Do not mix them: if a human should read it, it goes in `docs/`;
if it exists so that an agent can work correctly in the code, it goes here.

Read in this order before touching the simulator:

| File | Use |
|---|---|
| `CONVENTIONS.md` | working rules: frames, units, git identity, claims policy, where docs go, how to test |
| `SIM_ARCHITECTURE.md` | module map, data flow, APIs, how to add a controller / sensor / scenario |
| `POLICIES.md` | the four controllers, the guard and Maude's backup semantics, observation contracts |
| `KNOWN_ISSUES.md` | verified limitations and open findings (keep current; delete items when fixed) |
| `RECOVERY_SPEC.md` | spec of the proposed recovery layer (states, transitions, files, tests); status: proposal |

Human documents worth knowing: `docs/SIMULATOR_GUIDE.md`, `docs/SIM_TEST_REPORT.md`, `docs/SIMULATION_PLAN.md`,
`docs/ROBUSTNESS_PLAN.md`, `policy_integration/CONTRACT.md` (Maude's contract; authoritative for the policy interface).
