"""Verify warmup+constant LR schedule produces correct LR sequence."""
import torch


def _build_scheduler(lr: float, warmup_epochs: int, num_epochs: int):
    wu = warmup_epochs

    def _const_fn(ep: int) -> float:
        if wu <= 0 or ep >= wu:
            return 1.0
        return max(1e-6, ep / wu)

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_const_fn)
    return opt, sched


def test_constant_warmup_lr_sequence():
    """LR should ramp from ~0 over 5 epochs then stay constant at base_lr."""
    base_lr = 1e-4
    warmup = 5
    opt, sched = _build_scheduler(base_lr, warmup, 150)

    logged = []
    for epoch in range(1, 10):
        logged.append((epoch, opt.param_groups[0]["lr"]))
        sched.step()

    # epoch 1: near-zero warmup (lambda(0) = 1e-6)
    assert logged[0][1] < 1e-8, f"epoch 1 lr should be ~0, got {logged[0][1]}"
    # epochs 2-5: ramping up
    for ep, lr in logged[1:5]:
        assert 0 < lr < base_lr, f"epoch {ep} lr={lr} should be between 0 and {base_lr}"
    # epoch 6+: full lr
    for ep, lr in logged[5:]:
        assert abs(lr - base_lr) < 1e-10, f"epoch {ep} lr={lr} should equal {base_lr}"
    # monotonically increasing during warmup
    warmup_lrs = [lr for _, lr in logged[:5]]
    assert warmup_lrs == sorted(warmup_lrs), "warmup should be monotonically increasing"


def test_no_nan_during_warmup():
    """LR should never exceed base_lr."""
    base_lr = 1e-4
    opt, sched = _build_scheduler(base_lr, 5, 150)
    for _ in range(20):
        lr = opt.param_groups[0]["lr"]
        assert lr <= base_lr + 1e-12, f"LR {lr} exceeded base_lr {base_lr}"
        sched.step()
