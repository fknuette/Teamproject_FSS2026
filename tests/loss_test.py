import torch
from types import SimpleNamespace
import sys
from pathlib import Path
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]        # Teamproject_FSS2026
sys.path.insert(0, str(ROOT / "scripts"))          # -> grpo_training
sys.path.insert(0, str(ROOT / "src"))              # -> teamproject_fss2026


from grpo_training.loss import grpo_loss

CLIP_EPS = 0.2


class FixedLogitsModel(torch.nn.Module):
    """Gibt unabhängig vom Input feste Logits zurück -> Ratio exakt steuerbar."""
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone().float())

    def forward(self, input_ids=None, attention_mask=None, **kw):
        return SimpleNamespace(logits=self.logits)


def _make_batch(T=6, V=10, prompt_len=3):
    torch.manual_seed(0)
    input_ids = torch.randint(0, V, (1, T))
    attention_mask = torch.ones(1, T, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100          # Prompt-Tokens werden nicht gelernt
    return input_ids, attention_mask, labels


def test_identity_ratio_is_one():
    """Policy == Old == Ref  ->  ratio=1, kl=0, policy_loss=-advantage."""
    ids, am, labels = _make_batch()
    logits = torch.randn(1, ids.shape[1], 10)
    model = FixedLogitsModel(logits)          # dasselbe Modell für alle drei
    adv = torch.tensor([2.0])

    loss, m = grpo_loss(model, model, model, ids, am, labels, adv,
                        clip_epsilon=CLIP_EPS)

    assert abs(m["ratio_mean"] - 1.0) < 1e-5
    assert abs(m["kl_div"]) < 1e-5
    assert torch.allclose(loss, torch.tensor(-2.0), atol=1e-5)   # -advantage


def test_clipping_caps_positive_advantage():
    ids, am, labels = _make_batch()
    T = ids.shape[1]
    base = torch.randn(1, T, 10)
    old = FixedLogitsModel(base)

    def boost(delta):
        b = base.clone()
        for pos in range(T - 1):
            b[0, pos, ids[0, pos + 1]] += delta   # NÄCHSTES Token (next-token prediction!)
        return FixedLogitsModel(b)

    adv = torch.tensor([2.0])
    # kl_coef=0 -> Gesamt-Loss == Policy-Loss, damit isoliert testbar
    loss20, m20 = grpo_loss(boost(20.0), old, old, ids, am, labels, adv,
                            clip_epsilon=CLIP_EPS, kl_coef=0.0)
    loss40, _ = grpo_loss(boost(40.0), old, old, ids, am, labels, adv,
                          clip_epsilon=CLIP_EPS, kl_coef=0.0)

    assert m20["ratio_mean"] > 1.0 + CLIP_EPS          # Ratio wirklich außerhalb [0.8, 1.2]
    expected = torch.tensor(-2.0 * (1.0 + CLIP_EPS))   # -adv * 1.2
    assert torch.allclose(loss20, expected, atol=1e-4)
    assert torch.allclose(loss40, expected, atol=1e-4) # stärker geboostet -> unverändert


def test_negative_advantage_is_not_capped_but_finite():
    ids, am, labels = _make_batch()
    T = ids.shape[1]
    base = torch.randn(1, T, 10)
    old = FixedLogitsModel(base)
    b = base.clone()
    for pos in range(T - 1):
        b[0, pos, ids[0, pos + 1]] += 20.0
    policy = FixedLogitsModel(b)

    adv = torch.tensor([-2.0])
    loss, _ = grpo_loss(policy, old, old, ids, am, labels, adv,
                        clip_epsilon=CLIP_EPS, kl_coef=0.0)

    assert torch.isfinite(loss)
    assert loss.item() > 2.0 * (1.0 + CLIP_EPS)   # NICHT gedeckelt (pessimistische Seite)       # NICHT gedeckelt


def test_masked_tokens_do_not_affect_loss():
    """Logits an rein maskierten Positionen ändern den Loss nicht."""
    ids, am, labels = _make_batch()
    logits = torch.randn(1, ids.shape[1], 10)
    model_a = FixedLogitsModel(logits)
    adv = torch.tensor([1.0])
    loss_a, _ = grpo_loss(model_a, model_a, model_a, ids, am, labels, adv, clip_epsilon=CLIP_EPS)

    perturbed = logits.clone()
    perturbed[0, 0, :] += 100.0        # Position 0 sagt ein maskiertes Label voraus
    model_b = FixedLogitsModel(perturbed)
    loss_b, _ = grpo_loss(model_b, model_b, model_b, ids, am, labels, adv, clip_epsilon=CLIP_EPS)

    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_gradient_flows_only_to_policy():
    """backward() erzeugt endliche Gradienten an der Policy, keine an old/ref."""
    ids, am, labels = _make_batch()
    logits = torch.randn(1, ids.shape[1], 10)
    policy = FixedLogitsModel(logits + 0.1)
    old = FixedLogitsModel(logits)
    ref = FixedLogitsModel(logits)
    adv = torch.tensor([1.5])

    loss, _ = grpo_loss(policy, old, ref, ids, am, labels, adv, clip_epsilon=CLIP_EPS)
    loss.backward()

    assert policy.logits.grad is not None
    assert torch.isfinite(policy.logits.grad).all()
    assert old.logits.grad is None      # unter no_grad berechnet
    assert ref.logits.grad is None