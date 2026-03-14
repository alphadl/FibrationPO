# Reproducibility checklist — Fibration Policy Optimization (arXiv:2603.08239)

Not an official implementation. This repo reproduces the paper’s frameworks and formulas.

| Item | Paper reference | Module |
|------|-----------------|--------|
| RGF | Def 3.1, Eq (2): J^ = sum μ G(r)_{s,a,I} A^ | `fiberpo/rgf.py` |
| APC-Obj | Def 3.2, Eq (3); Thm 3.3 (equiv. TV-TRPO) | `fiberpo/apc_obj.py` |
| FBG | Sec 4: base g^agg(δ), fiber logclip(ε); Thm 4.5 | `fiberpo/fbg.py` |
| FiberPO | Two-level (trajectory, token); block-diagonal Jacobian | `fiberpo/fiber_po.py` |
| FiberPO-Domain | Four-level FGH (domain, prompt_group, trajectory, token) | `fiberpo/fiber_po_domain.py` |

**Formulas:**  
- Eq (3): clip(r_sa−1, B_sa) with B_sa = T_s δ − sum_{other} |r−1|; clip(a,B)=clip(a,−B+,B+), B+=max(B,0).  
- FBG: pushforward → g^agg(delta) → reflect → logclip(epsilon).  
- GSPO scale (Remark 3.6): ε ≈ 4e-4 ⇒ δ ~ 0.014; FiberPO defaults δ=ε=0.2 for stability.
