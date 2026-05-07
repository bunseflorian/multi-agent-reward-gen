Improvement: 
Get dones still terminates on boundary violation -> Negative terms dont work, because the drone can just crash
Can the planer extract this information and give it to the coder?
If termination on crash:
    Main objective must be positive
    Negative-only terms are unsafe, because the agent can terminate early by crashing and avoid future penalties
If no termination on crash:
    Negative constraint terms are admissible, but only if they cannot be satisfied in degenerate non-task states
    Constraint shaping should be gated by task progress, valid task execution, or goal proximity
    Otherwise the agent may exploit low-effort failure modes, e.g. lying on the ground with low angular velocity