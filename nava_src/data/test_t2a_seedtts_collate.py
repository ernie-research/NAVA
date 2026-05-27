"""
Pytest unit tests for collate_fn in t2a_seedtts.py
"""
import pytest
import torch
from NAVA.nava_src.data.t2a_seedtts import collate_fn


class TestCollateFn:

    # ── helpers ──────────────────────────────────────────────────────────
    def _make_sample(
        self,
        idx=0,
        utt_id="u1",
        prompt_text="hello",
        prompt_audio_path="/tmp/a.wav",
        captions="world",
        spk_embs=None,
        audio_latents=None,
        save_path="/tmp/s.wav",
    ):
        return {
            "idx": idx,
            "utt_id": utt_id,
            "prompt_text": prompt_text,
            "prompt_audio_path": prompt_audio_path,
            "captions": captions,
            "spk_embs": spk_embs,
            "audio_latents": audio_latents,
            "save_path": save_path,
        }

    # ── normal cases ────────────────────────────────────────────────────
    def test_normal_batch_all_keys_present_audio_latents_tensors(self):
        """All keys present, audio_latents are tensors → audio_seq_len from shape."""
        latent1 = torch.randn(50, 20)
        latent2 = torch.randn(30, 20)
        batch = [
            self._make_sample(idx=0, audio_latents=latent1, spk_embs=torch.randn(1, 192)),
            self._make_sample(idx=1, audio_latents=latent2, spk_embs=torch.randn(1, 192)),
        ]
        out = collate_fn(batch)

        assert out["idx"] == [0, 1]
        assert out["utt_id"] == ["u1", "u1"]
        assert out["prompt_text"] == ["hello", "hello"]
        assert out["prompt_audio_path"] == ["/tmp/a.wav", "/tmp/a.wav"]
        assert out["captions"] == ["world", "world"]
        assert out["save_path"] == ["/tmp/s.wav", "/tmp/s.wav"]
        assert out["audio_seq_len"] == [50, 30]
        assert out["t_h_w_list"] is None
        assert len(out["spk_embs"]) == 2
        assert out["audio_latents"] is not None

    def test_single_item_batch(self):
        """Single item in batch."""
        latent = torch.randn(40, 20)
        batch = [self._make_sample(idx=5, audio_latents=latent)]
        out = collate_fn(batch)

        assert out["idx"] == [5]
        assert out["audio_seq_len"] == [40]
        assert out["t_h_w_list"] is None

    # ── empty batch ─────────────────────────────────────────────────────
    def test_empty_batch(self):
        """Empty batch → all vals are []→None or [0]*0."""
        out = collate_fn([])

        # Every processed key → vals = [] → all(x is None for x in []) is True → vals = None
        assert out["idx"] is None
        assert out["utt_id"] is None
        assert out["prompt_text"] is None
        assert out["prompt_audio_path"] is None
        assert out["captions"] is None
        assert out["spk_embs"] is None
        assert out["audio_latents"] is None
        assert out["save_path"] is None
        # "audio_latents" in out and out["audio_latents"] is None → else branch
        assert out["audio_seq_len"] == []
        assert out["t_h_w_list"] is None

    # ── all values None for a key ───────────────────────────────────────
    def test_all_values_none_for_key_sets_val_to_none(self):
        """All items have None for spk_embs → out[\"spk_embs\"] set to None."""
        latent1 = torch.randn(10, 20)
        batch = [
            self._make_sample(idx=0, spk_embs=None, audio_latents=latent1),
            self._make_sample(idx=1, spk_embs=None, audio_latents=latent1),
        ]
        out = collate_fn(batch)

        assert out["spk_embs"] is None
        # Other keys that are not all None remain as lists
        assert out["idx"] == [0, 1]

    def test_all_values_none_for_multiple_keys(self):
        """Multiple keys where all values are None → each set to None."""
        latent = torch.randn(5, 20)
        batch = [
            self._make_sample(idx=0, spk_embs=None, save_path=None, audio_latents=latent),
            self._make_sample(idx=1, spk_embs=None, save_path=None, audio_latents=latent),
        ]
        out = collate_fn(batch)

        assert out["spk_embs"] is None
        assert out["save_path"] is None
        assert out["idx"] == [0, 1]
        assert out["audio_seq_len"] == [5, 5]

    # ── mixed None/non-None for a key ───────────────────────────────────
    def test_mixed_none_and_non_none_keeps_list(self):
        """Some items have None, some have a value → keeps the list."""
        spk1 = torch.randn(1, 192)
        latent = torch.randn(10, 20)
        batch = [
            self._make_sample(idx=0, spk_embs=spk1, audio_latents=latent),
            self._make_sample(idx=1, spk_embs=None, audio_latents=latent),
        ]
        out = collate_fn(batch)

        assert out["spk_embs"] is not None
        assert len(out["spk_embs"]) == 2
        assert out["spk_embs"][0] is spk1
        assert out["spk_embs"][1] is None

    # ── audio_latents branches ──────────────────────────────────────────
    def test_audio_latents_present_but_out_value_is_none(self):
        """audio_latents key in batch but all items have None → out None → else branch."""
        batch = [
            self._make_sample(idx=0, audio_latents=None),
            self._make_sample(idx=1, audio_latents=None),
            self._make_sample(idx=2, audio_latents=None),
        ]
        out = collate_fn(batch)

        # "audio_latents" in out is True but out["audio_latents"] is None (falsy) → else
        assert out["audio_latents"] is None
        assert out["audio_seq_len"] == [0, 0, 0]

    def test_audio_latents_present_some_items_have_none(self):
        """Some audio_latents are None in batch → uses 0 for those items."""
        latent = torch.randn(20, 20)
        batch = [
            self._make_sample(idx=0, audio_latents=latent),
            self._make_sample(idx=1, audio_latents=None),
            self._make_sample(idx=2, audio_latents=latent),
        ]
        out = collate_fn(batch)

        assert out["audio_seq_len"] == [20, 0, 20]

    # ── t_h_w_list ──────────────────────────────────────────────────────
    def test_t_h_w_list_always_none(self):
        """t_h_w_list is always set to None regardless of input."""
        latent = torch.randn(5, 20)
        batch = [self._make_sample(audio_latents=latent)]
        out = collate_fn(batch)
        assert out["t_h_w_list"] is None

        out_empty = collate_fn([])
        assert out_empty["t_h_w_list"] is None

    # ── keys that appear in some items but not all ──────────────────────
    def test_key_missing_in_some_items_returns_none_for_those(self):
        """
        A key is in processed_keys but some items don't have it.
        b.get(k, None) returns None for those items.
        Since not all are None (others have values), the list is kept.
        """
        latent = torch.randn(5, 20)
        item_with_save = self._make_sample(idx=0, audio_latents=latent, save_path="/p1.wav")
        item_without_save = {
            "idx": 1,
            "utt_id": "u2",
            "prompt_text": "hi",
            "prompt_audio_path": "/tmp/b.wav",
            "captions": "there",
            "spk_embs": None,
            "audio_latents": latent,
            # NO save_path
        }
        batch = [item_with_save, item_without_save]
        out = collate_fn(batch)

        assert out["save_path"] == ["/p1.wav", None]

    # ── different-shaped audio_latents ──────────────────────────────────
    def test_audio_latents_various_shapes(self):
        """Different audio_latent tensor shapes → correct seq lens recorded."""
        latent1 = torch.zeros(10, 20)
        latent2 = torch.zeros(100, 20)
        latent3 = torch.zeros(1, 20)
        batch = [
            self._make_sample(idx=0, audio_latents=latent1),
            self._make_sample(idx=1, audio_latents=latent2),
            self._make_sample(idx=2, audio_latents=latent3),
        ]
        out = collate_fn(batch)
        assert out["audio_seq_len"] == [10, 100, 1]

    # ── large batch ─────────────────────────────────────────────────────
    def test_large_batch(self):
        """Correctness with many samples."""
        n = 100
        batch = [
            self._make_sample(idx=i, audio_latents=torch.randn(i + 1, 20))
            for i in range(n)
        ]
        out = collate_fn(batch)
        assert len(out["idx"]) == n
        assert out["audio_seq_len"] == [i + 1 for i in range(n)]
        assert out["t_h_w_list"] is None

    # ── keys with non-standard types ────────────────────────────────────
    def test_keys_with_various_data_types(self):
        """int, str, None, tensor types in values."""
        latent = torch.randn(10, 20)
        batch = [
            {
                "idx": 42,
                "utt_id": "utt-001",
                "prompt_text": "hello world",
                "prompt_audio_path": "/data/ref.wav",
                "captions": "target text",
                "spk_embs": torch.randn(1, 192),
                "audio_latents": latent,
                "save_path": "/out/u.wav",
            }
        ]
        out = collate_fn(batch)
        assert out["idx"] == [42]
        assert isinstance(out["idx"][0], int)
        assert isinstance(out["utt_id"][0], str)
        assert isinstance(out["captions"][0], str)
        assert out["audio_seq_len"] == [10]