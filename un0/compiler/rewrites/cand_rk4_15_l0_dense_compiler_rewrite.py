from un0.compiler.registry import register_compiler_rewrite
from un0.compiler.functional_hop import functionalize_kuramoto_model

@register_compiler_rewrite("cand_rk4_15_l0_dense_compiler_rewrite")
def cand_rk4_15_l0_dense_compiler_rewrite_fn(model, solver_name="rk4", num_steps=45, dt=0.066667, integration_time=1.0, target_device="cuda", **kwargs):
    """AOT compilable functional rewrite for cand_rk4_15_l0_dense."""
    return functionalize_kuramoto_model(
        model=model,
        solver_name=solver_name,
        num_steps=num_steps,
        dt=dt,
        integration_time=integration_time,
        use_hop=False,
    )

cand_rk4_15_l0_dense_compiler_rewrite = cand_rk4_15_l0_dense_compiler_rewrite_fn
rewrite = cand_rk4_15_l0_dense_compiler_rewrite_fn
rewrite_fn = cand_rk4_15_l0_dense_compiler_rewrite_fn
