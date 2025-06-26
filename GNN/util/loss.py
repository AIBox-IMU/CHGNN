import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, probs, targets):
        """
        输入要求：
        - probs: 经过sigmoid的概率值 (范围[0,1])
        - targets: 与probs同形状的标签
        """
        # 计算二元交叉熵（不使用logits版本）
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        # 计算概率调整因子
        pt = torch.where(targets == 1, probs, 1 - probs)

        # 计算alpha权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 计算Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# 使用示例
if __name__ == "__main__":
    # 假设模型输出已经是概率值
    probs = torch.sigmoid(torch.randn(4, 1))  # 模拟sigmoid输出
    targets = torch.tensor([[1.0], [0.0], [1.0], [1.0]])

    # 原始BCE Loss
    criterion_bce = nn.BCELoss()
    loss_bce = criterion_bce(probs, targets)
    print(f"BCE Loss: {loss_bce.item()}")

    # Focal Loss
    criterion_focal = FocalLoss(alpha=0.25, gamma=2)
    loss_focal = criterion_focal(probs, targets)
    print(f"Focal Loss: {loss_focal.item()}")