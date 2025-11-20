import torch

from .basis_gn import UniBSplineBasis
from .uni_bspline import UniformBSpline


class SplineFactory:

    @staticmethod
    def init_splines(mp_type: str,
                mp_args: dict,
                num_dof: int = 1,
                tau: float = 1,
                dtype: torch.dtype = torch.float32,
                device: torch.device = "cpu"):

        if mp_type == "uni_bspline":
            basis_gn = UniBSplineBasis(dtype=dtype, device=device, tau=tau,
                                       **mp_args)
            mp = UniformBSpline(basis_gn=basis_gn, num_dof=num_dof,
                                dtype=dtype, device=device, **mp_args)
        else:
            raise NotImplementedError

        return mp