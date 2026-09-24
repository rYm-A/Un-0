from un0.compiler.registry import register_compiler_rewrite
from un0.compiler.functional_hop import functionalize_kuramoto_model

@register_compiler_rewrite("cand_eb15_l2_dense_compiler_rewrite")
@register_compiler_rewrite("cand_eb15_l2_dense")
@register_compiler_rewrite("compiler_rewrite")
def cand_eb15_l2_dense_compiler_rewrite_fn(
    model,
    solver_name="euler_backward",
    num_steps=15,
    dt=0.066667,
    integration_time=1.0,
    target_device="cuda",
    **kwargs
):
    """AOT compilable functional rewrite for cand_eb15_l2_dense."""
    actual_steps = 15 if (num_steps is None or num_steps == 45) else num_steps
    actual_dt = 0.066667 if (dt is None) else dt
    return functionalize_kuramoto_model(
        model=model,
        solver_name=solver_name,
        num_steps=actual_steps,
        dt=actual_dt,
        integration_time=integration_time,
        use_hop=False,
    )

rewrite = cand_eb15_l2_dense_compiler_rewrite_fn
rewrite_fn = cand_eb15_l2_dense_compiler_rewrite_fn
cand_eb15_l2_dense_compiler_rewrite = cand_eb15_l2_dense_compiler_rewrite_fn
cand_eb15_l2_dense = cand_eb15_l2_dense_compiler_rewrite_fn
compiler_rewrite = cand_eb15_l2_dense_compiler_rewrite_fn
