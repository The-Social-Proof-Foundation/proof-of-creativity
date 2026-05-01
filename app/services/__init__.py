"""
Media processing services (embedding, fingerprinting, video processing).

Heavy modules are imported from their concrete submodules — not re-exported here —
so importing lightweight helpers does not pull in optional GPU/torch stacks.
"""
