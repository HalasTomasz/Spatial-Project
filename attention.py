import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MyAttention(nn.Module):
    def __init__(self, device):
        super(MyAttention, self).__init__()
        self.device = device

    def forward(self, generated, known, mask):
        return MyAttentionFunction.apply(generated, known, mask, self.device)


class MyAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, generated, known, mask, device):
        k = 2  # Number of neighbors to consider
        ctx.k = k

        # Process mask and interpolate to match dimensions
        ctx.flag = F.interpolate(mask.clone(), (32, 32)).view(1, -1).to(device)

        # Extract patches from `generated` and `known`
        patches_all = MyUnfold(generated, 1, 1)
        patches = patches_all[ctx.flag == 1].view(1, -1, patches_all.size(1))

        known_patches_all = MyUnfold(known, 1, 1)
        known_patches = known_patches_all[ctx.flag == 0].view(1, -1, known_patches_all.size(1))

        # Compute cosine similarities
        num = torch.einsum('bik,bjk->bij', patches, known_patches)
        norm_patches = torch.norm(patches, dim=2)
        norm_known = torch.norm(known_patches, dim=2)
        den = torch.einsum('bi,bj->bij', norm_patches, norm_known)

        cosine0 = num / den
        cosine1 = torch.einsum('bik,bjk->bij', patches, patches) / torch.einsum('bi,bj->bij', norm_patches, norm_patches)

        # Compute weights using softmax over top-k similarities
        weight0, indexes0 = torch.topk(cosine0, k, dim=2)
        weight1, indexes1 = torch.topk(cosine1, k, dim=2)
        
        ctx.weights = F.softmax(torch.cat((weight0.mean(dim=2), weight1.mean(dim=2)), dim=1), dim=1).tolist()

        # Prepare indices for backward pass
        mask_indexes = (ctx.flag == 1).nonzero(as_tuple=True)
        ctx.ind = []
        
        for i in range(k):
            ind_mask0 = torch.zeros_like(ctx.flag, dtype=torch.float32).to(device)
            ind_mask0[mask_indexes] = known_patches_all[indexes0[:, :, i]].reshape(-1)
            ctx.ind.append(ind_mask0)

        # Combine contributions
        rtn = sum(
            torch.bmm(ind.unsqueeze(0), patches_all if i >= k else known_patches_all) * ctx.weights[i]
            for i, ind in enumerate(ctx.ind)
        )

        rtn = rtn.view(1, 32, 32, known.size(1)).permute(0, 3, 1, 2)
        return torch.cat([generated, known, rtn], dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        c = grad_output.size(1)
        grad_former_all = grad_output[:, :c // 3, :, :]
        grad_latter_all = grad_output[:, c // 3:2 * c // 3, :, :]
        grad_shifted_all = grad_output[:, 2 * c // 3:, :, :]

        for i, ind in enumerate(ctx.ind):
            W_mat_t = ind.permute(0, 2, 1).contiguous()
            grad = grad_shifted_all.view(1, c // 3, -1).permute(0, 2, 1)
            grad_shifted_weighted = torch.bmm(W_mat_t, grad)
            grad_latter_all += grad_shifted_weighted.permute(0, 2, 1).view(1, c // 3, 32, 32)

        return grad_former_all, grad_latter_all, None, None


def MyUnfold(input, patch_size, stride):
    """Extract patches from input tensor."""
    patches = input.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(patches.size(0), -1, patches.size(3))
