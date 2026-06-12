import torch

from net.roi_pool import ROITopKPooling


def test_roi_pool_returns_all_standardized_scores_and_half_nodes():
    pool = ROITopKPooling(2, ratio=0.5)
    with torch.no_grad():
        pool.weight.copy_(torch.tensor([[1.0, 0.0]]))
    x = torch.tensor(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0],
         [10.0, 0.0], [11.0, 0.0], [13.0, 0.0], [16.0, 0.0]]
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    pooled_x, _, _, pooled_batch, perm, selected, scores = pool(x, edge_index, batch=batch)
    assert pooled_x.shape[0] == 4
    assert selected.shape[0] == 4
    assert scores.shape[0] == 8
    assert torch.equal(torch.bincount(pooled_batch), torch.tensor([2, 2]))
    assert set(perm[:2].tolist()) == {2, 3}
    assert set(perm[2:].tolist()) == {6, 7}


def test_roi_pool_scores_are_invariant_to_per_graph_shift_and_scale():
    pool = ROITopKPooling(1, ratio=0.5)
    with torch.no_grad():
        pool.weight.fill_(1.0)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [10.0], [12.0], [14.0], [16.0]])
    scores = pool(x, edge_index, batch=batch)[-1]
    torch.testing.assert_close(scores[:4], scores[4:], atol=1e-6, rtol=1e-6)
