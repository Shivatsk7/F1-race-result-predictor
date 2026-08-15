"""Feature registry and leakage firewall. Implements FR-201..208.

The registry is load-bearing: features declare when their inputs become
available, and the primary feature set is derived from that declaration
rather than hand-maintained. See plan.md D-1.
"""
