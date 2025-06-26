import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .complexhyperbolic import chyp_distance, expmap0, logmap0, project, mobius_add, Distance, real_mobius_add
from .euclidean import givens_rotations, givens_reflection
from .complexhyperbolic import mobius_mul,logmap_0

class FFTAttH(nn.Module):

    def __init__(self,params):
        super(FFTAttH, self).__init__()
        self.params = params
        self.c = nn.Embedding(self.params.num_rels, 1)
        self.rank = self.params.emb_dim // 2
        self.dim = 2 * (self.rank - 1)

        self.rel = nn.Embedding(self.params.num_rels, 2*self.dim)
        self.rel_diag = nn.Embedding(self.params.num_rels, self.dim)
        self.context_vec = nn.Embedding(self.params.num_rels, self.dim)
        self.act = nn.Softmax(dim=-2)
        self.scale = 1. / np.sqrt(self.rank)
        self.init_size = 1e-3
        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0.0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.ones_(self.c.weight)

    def att_get_queries(self, node, rel_labels):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(rel_labels))
        head = node
        head = head[..., :self.rank] + 1j * head[..., self.rank:]
        head = torch.fft.irfft(head, norm="ortho")
        rot_mat, ref_mat = torch.chunk(self.rel(rel_labels), 2, dim=-1)
        rot_q = givens_rotations(rot_mat, head).unsqueeze(-2)
        ref_q = givens_reflection(ref_mat, head).unsqueeze(-2)
        cands = torch.cat([ref_q, rot_q], dim=-2)
        context_vec = self.context_vec(rel_labels).unsqueeze(-2)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=-2)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(rel_labels), 2, dim=-1)
        rel = expmap0(rel, c)
        res = project(real_mobius_add(lhs, rel, c), c)
        res = torch.fft.rfft(res, norm="ortho")
        res = torch.cat((res.real, res.imag), -1)
        while res.dim() < 3:
            res = res.unsqueeze(1)
        return res