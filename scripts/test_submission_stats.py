"""Checks mean versus median and custom token counters for NB2 reports."""
import pytest
from lab22 import data as D

def row(chosen,rejected):
    return D.to_conversational({"prompt":"example", "chosen":chosen, "rejected":rejected})

def test_means_are_distinct_from_medians():
    stats=D.length_stats([row("a","abcdef"),row("ab","abc"),row("abcdefghi","abc")])
    assert stats["chosen_mean"]==pytest.approx(4.0)
    assert stats["rejected_mean"]==pytest.approx(4.0)
    assert stats["chosen_median"]==2.0
    assert stats["rejected_median"]==3.0
    assert stats["chosen_longer_frac"]==pytest.approx(1/3)

def test_means_use_supplied_token_counter():
    stats=D.length_stats([row("a","a b c d e f"),row("b c","g h i"),row("d e f g h i j k l","j k l")],count=lambda text:len(text.split()))
    assert stats["chosen_mean"]==pytest.approx(4.0)
    assert stats["rejected_mean"]==pytest.approx(4.0)
    assert stats["chosen_median"]==2.0
    assert D.length_stats([])=={"n":0}


def test_student_dpo_loss_matches_reference_and_preserves_policy_gradients():
    import ast
    from pathlib import Path
    import torch
    from lab22 import dpo_math as M
    source=Path(__file__).resolve().parent.parent/'notebooks/00_dpo_loss_from_scratch.py'
    tree=ast.parse(source.read_text())
    func=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="my_dpo_loss")
    ns={"torch":torch}
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),"exec"),ns)
    pc=torch.tensor([-12.0,-20.0],requires_grad=True)
    pr=torch.tensor([-15.0,-19.0],requires_grad=True)
    rc=torch.tensor([-13.0,-21.0])
    rr=torch.tensor([-14.0,-20.0])
    loss=ns["my_dpo_loss"](pc,pr,rc,rr,beta=0.3)
    expected,_,_=M.dpo_loss(pc,pr,rc,rr,beta=0.3)
    assert torch.allclose(loss,expected,atol=1e-6)
    loss.backward()
    assert pc.grad is not None and pr.grad is not None
    assert torch.isfinite(pc.grad).all() and torch.isfinite(pr.grad).all()
    assert (pc.grad<0).all() and (pr.grad>0).all()
