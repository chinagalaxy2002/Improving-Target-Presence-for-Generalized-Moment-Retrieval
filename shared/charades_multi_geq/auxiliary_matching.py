"""Keep SciPy Hungarian assignments while transferring a whole layer together."""
import torch
from scipy.optimize import linear_sum_assignment


def hungarian_batch(costs):
    """One device-to-host transfer, with unchanged per-example costs and solver.

    Matrices may have different target counts. Packing only copies their values;
    in particular it does not replace Hungarian tie-breaking with top-k.
    """
    if not costs:
        return []
    first = costs[0]
    assert all(c.ndim == 2 and c.dtype == first.dtype and c.device == first.device
               for c in costs)
    host = torch.cat([c.detach().reshape(-1) for c in costs]).cpu().numpy()
    assignments = []
    offset = 0
    for cost in costs:
        count = cost.numel()
        assignments.append(linear_sum_assignment(host[offset:offset + count].reshape(cost.shape)))
        offset += count
    return assignments
