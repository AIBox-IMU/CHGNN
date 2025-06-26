import torch
import torch.nn.functional as F
from utils.complexhyperbolic import Distance


def neg_sampling_loss(positive_score,negative_score):
    """Compute KG embedding loss with negative sampling.

    Args:
        input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

    Returns:
        loss: torch.Tensor with negative sampling embedding loss
        factors: torch.Tensor with embeddings weights to regularize
    """
    # positive samples
    positive_score = F.logsigmoid(positive_score)

    # negative samples
    negative_score = F.logsigmoid(-negative_score)
    loss = - torch.cat([positive_score.view(-1), negative_score.view(-1)]).mean()
    return loss


def similarity_score(lhs_e, rhs_e):
    """Compute similarity scores or queries against targets in embedding space.

    Returns:
        scores: torch.Tensor with similarity scores of queries against targets
    """
    return - Distance.apply(lhs_e, rhs_e) ** 2