"""Regression tests for the position_id_policy axis (padding/position-ID diagnostic).

Covers:
  * _maybe_position_ids unit behavior (left/right padding, default, unknown policy)
  * all three model-wrapper forwards accept position_ids (backward-compatible)
  * mechanism: mask-derived IDs CHANGE GPT-2 logits for left-padded input whose
    padded length differs from natural length
  * control: mask-derived IDs DO NOT change BERT logits for right-padded input
  * training-path regression: forward without position_ids == forward with None

CPU-only, tiny 'small' configs, deterministic seeds. No GPU, no network.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from pls_run_inference_context import _maybe_position_ids
    from pls_train_classifier import BertModule as BERTModule, GPTModule, MLMModule
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"model-stack dependency unavailable: {error.name}") from error


class TestMaybePositionIds(unittest.TestCase):
    def test_left_padded(self):
        am = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
        pos = _maybe_position_ids({"attention_mask": am}, "mask_derived")
        expected = torch.tensor([[0, 0, 0, 0, 1, 2], [0, 0, 1, 2, 3, 4]])
        self.assertTrue(torch.equal(pos, expected))

    def test_right_padded_real_tokens_are_arange(self):
        am = torch.tensor([[1, 1, 1, 0, 0]])
        pos = _maybe_position_ids({"attention_mask": am}, "mask_derived")
        # real tokens get 0..L-1 : identical to default arange at those positions
        self.assertTrue(torch.equal(pos[0, :3], torch.arange(3)))

    def test_default_returns_none(self):
        am = torch.ones(2, 4, dtype=torch.long)
        self.assertIsNone(_maybe_position_ids({"attention_mask": am}, "default"))
        self.assertIsNone(_maybe_position_ids({"attention_mask": am}, None))

    def test_unknown_policy_raises(self):
        am = torch.ones(1, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            _maybe_position_ids({"attention_mask": am}, "bogus")


def _left_padded_batch(vocab_size, batch=2, total_len=16, real_len=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.zeros(batch, total_len, dtype=torch.long)
    attention_mask = torch.zeros(batch, total_len, dtype=torch.long)
    real = torch.randint(1, vocab_size - 10, (batch, real_len), generator=g)
    input_ids[:, -real_len:] = real
    attention_mask[:, -real_len:] = 1
    return input_ids, attention_mask


def _right_padded_batch(vocab_size, batch=2, total_len=16, real_len=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.zeros(batch, total_len, dtype=torch.long)
    attention_mask = torch.zeros(batch, total_len, dtype=torch.long)
    real = torch.randint(1, vocab_size - 10, (batch, real_len), generator=g)
    input_ids[:, :real_len] = real
    attention_mask[:, :real_len] = 1
    return input_ids, attention_mask


class TestForwardsThreading(unittest.TestCase):
    """Random-weight tiny models; deterministic; eval mode; no_grad."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        cls.gpt_pure = GPTModule("small", "pure-gen-GPT2", labels=[0, 1],
                                 label_token_ids=[50257 - 3, 50257 - 2]).eval()
        torch.manual_seed(1234)
        cls.gpt_pseudo = GPTModule("small", "pseudo-gen-GPT2", labels=[0, 1],
                                   label_token_ids=[50257 - 3, 50257 - 2]).eval()
        torch.manual_seed(1234)
        cls.bert = BERTModule("small", labels=[0, 1]).eval()
        torch.manual_seed(1234)
        cls.mlm = MLMModule("small", "pseudo-gen-MLM", labels=[0, 1],
                            label_token_ids=[100, 101]).eval()

    def _logits(self, model, input_ids, attention_mask, position_ids=None):
        with torch.no_grad():
            if position_ids is None:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids)
        return out.raw_logits

    def test_gpt_pure_mask_derived_changes_logits_left_padded(self):
        ids, am = _left_padded_batch(self.gpt_pure.gpt2.config.vocab_size)
        pos = _maybe_position_ids({"attention_mask": am}, "mask_derived")
        base = self._logits(self.gpt_pure, ids, am)
        alt = self._logits(self.gpt_pure, ids, am, pos)
        self.assertGreater((base - alt).abs().max().item(), 1e-4,
                           "mask_derived must change left-padded GPT-2 logits")

    def test_gpt_pseudo_sliced_path_threads_position_ids(self):
        ids, am = _left_padded_batch(self.gpt_pseudo.gpt2.config.vocab_size)
        pos = _maybe_position_ids({"attention_mask": am}, "mask_derived")
        base = self._logits(self.gpt_pseudo, ids, am)
        alt = self._logits(self.gpt_pseudo, ids, am, pos)  # exercises [:, :-1] slice
        self.assertEqual(alt.shape, base.shape)
        self.assertGreater((base - alt).abs().max().item(), 1e-4)

    def test_bert_mask_derived_is_noop_right_padded(self):
        ids, am = _right_padded_batch(self.bert.model.config.vocab_size)
        pos = _maybe_position_ids({"attention_mask": am}, "mask_derived")
        base = self._logits(self.bert, ids, am)
        alt = self._logits(self.bert, ids, am, pos)
        self.assertTrue(torch.allclose(base, alt, atol=1e-5),
                        "mask_derived must be a no-op for right-padded BERT")

    def test_mlm_accepts_position_ids(self):
        ids, am = _right_padded_batch(self.mlm.bert.config.vocab_size)
        pos = _maybe_position_ids({"attention_mask": am}, "mask_derived")
        base = self._logits(self.mlm, ids, am)
        alt = self._logits(self.mlm, ids, am, pos)
        self.assertEqual(alt.shape, base.shape)
        self.assertTrue(torch.allclose(base, alt, atol=1e-5))

    def test_backward_compat_none_equals_omitted(self):
        """Training path regression: not passing position_ids == passing None."""
        for model, batch_fn, vocab in (
            (self.gpt_pure, _left_padded_batch, self.gpt_pure.gpt2.config.vocab_size),
            (self.bert, _right_padded_batch, self.bert.model.config.vocab_size),
        ):
            ids, am = batch_fn(vocab)
            base = self._logits(model, ids, am)
            explicit_none = self._logits(model, ids, am, None)
            self.assertTrue(torch.allclose(base, explicit_none, atol=0.0),
                            f"{type(model).__name__}: None must equal omitted")


if __name__ == "__main__":
    unittest.main()


class TestConsistencyTrainingPolicy(unittest.TestCase):
    """position_id_policy=mask_derived as a CLASS-INTERNAL default (consistency
    training), plus the explicit arange override used by the diagnostic."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(99)
        cls.gpt_default = GPTModule("small", "pure-gen-GPT2", labels=[0, 1],
                                    label_token_ids=[50257 - 3, 50257 - 2]).eval()
        torch.manual_seed(99)
        cls.gpt_maskpos = GPTModule("small", "pure-gen-GPT2", labels=[0, 1],
                                    label_token_ids=[50257 - 3, 50257 - 2],
                                    position_id_policy="mask_derived").eval()

    def test_internal_mask_derived_equals_explicit(self):
        ids, am = _left_padded_batch(self.gpt_maskpos.gpt2.config.vocab_size)
        pos = _maybe_position_ids({"attention_mask": am}, "mask_derived")
        with torch.no_grad():
            internal = self.gpt_maskpos(input_ids=ids, attention_mask=am).raw_logits
            explicit = self.gpt_default(input_ids=ids, attention_mask=am,
                                        position_ids=pos).raw_logits
        self.assertTrue(torch.allclose(internal, explicit, atol=1e-5),
                        "class-internal mask_derived must equal explicit pass")

    def test_explicit_override_beats_internal_policy(self):
        ids, am = _left_padded_batch(self.gpt_maskpos.gpt2.config.vocab_size)
        T = am.size(1)
        arange = torch.arange(T).unsqueeze(0).expand(am.size(0), -1)
        with torch.no_grad():
            overridden = self.gpt_maskpos(input_ids=ids, attention_mask=am,
                                          position_ids=arange).raw_logits
            plain_default = self.gpt_default(input_ids=ids, attention_mask=am).raw_logits
        self.assertTrue(torch.allclose(overridden, plain_default, atol=1e-5),
                        "explicit arange must override the internal policy")

    def test_arange_helper(self):
        am = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        pos = _maybe_position_ids({"attention_mask": am}, "arange")
        self.assertTrue(torch.equal(pos, torch.arange(4).unsqueeze(0).expand(2, -1)))
