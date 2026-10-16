# 0001. Remove the Azure Quantum and Key Vault integration

Status: accepted (Fase 0)

## Context

`src/quantum/azure_connector.py` never ran a circuit on Azure in any test or
demo. When the workspace was not connected, `_simulate_locally` ignored the
circuit it was given and ran a different chain of `RY` gates, so callers got
wrong results without an error (audit Q7). The Celery task
`sync_hardware_status` returned a hardcoded table of IonQ, Rigetti and
Quantinuum targets. `AzureKeyVaultManager` fell back to environment variables
whenever the vault was unreachable. Installing the declared dependencies failed
because `azure-mgmt-quantum>=1.0.0` has no stable release. None of the three
research questions in the project plan needs hardware access, and there is no
workspace to test against.

## Decision

Delete the connector, the Key Vault manager, the hardware-status task, the
`target_hardware` field of the quantum job API, the `azure-*`, `qiskit` and
`pennylane-qiskit` dependencies, the Azure variables in `.env.example` and
`docker-compose.yml`, and the Azure claims in the README. All circuits run on
PennyLane simulators.

## Consequences

Every quantum result in this repository comes from a simulator, and docs say
so. Hardware support, if it comes back, goes in a separate optional extra under
`qedge.quantum.backends.<provider>` that executes the tape it receives or
raises. It must not fall back to a local simulator.
