"""Regression tests for issue #49: SafetensorError in a DataLoader worker.

A transient read failure on a cached safetensors file (latent / text-embed)
inside a worker used to die in the worker's result queue (the SafetensorError
class is unpicklable), silently dropping the batch and hanging the run.

Covers:
  1. SafetensorError pickling unmask (class, instance, ExceptionWrapper)
  2. load_cached_tensors retry semantics (transient recovery, terminal error)
  3. get_latent through the real mixin (corrupt file, mid-retry repair)
  4. End-to-end: SafetensorError raised in a real DataLoader worker reaches
     the main process as a real error instead of hanging

Run with: python testing/test_cache_read_resilience.py
All tests run on CPU, no models required.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import tempfile

import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

import toolkit.dataloader_mixins as dm


def _run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ---------------------------------------------------------------------------
# 1. Pickling unmask
# ---------------------------------------------------------------------------

def test_safetensor_error_module_unmasked():
    assert SafetensorError.__module__ == 'safetensors', SafetensorError.__module__


def test_safetensor_error_class_and_instance_pickle():
    cls = pickle.loads(pickle.dumps(SafetensorError))
    assert cls is SafetensorError
    inst = pickle.loads(pickle.dumps(SafetensorError('boom')))
    assert isinstance(inst, SafetensorError)
    assert 'boom' in str(inst)


def test_exception_wrapper_round_trip():
    # The exact path from the incident: PyTorch wraps a worker exception in
    # ExceptionWrapper (which carries the exception class) and pickles it
    # onto the result queue.
    from torch._utils import ExceptionWrapper
    try:
        raise SafetensorError('Error while deserializing header')
    except Exception:
        w = ExceptionWrapper(where='in DataLoader worker process 0')
    w2 = pickle.loads(pickle.dumps(w))
    try:
        w2.reraise()
        assert False, 'reraise did not raise'
    except SafetensorError as e:
        assert 'deserializing header' in str(e)


# ---------------------------------------------------------------------------
# 2. load_cached_tensors retry semantics
# ---------------------------------------------------------------------------

class _NoSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def test_retry_recovers_from_transient_failure():
    no_sleep = _NoSleep()
    real_sleep = dm.time.sleep
    dm.time.sleep = no_sleep
    try:
        state = {'fails': 2}

        def flaky():
            if state['fails'] > 0:
                state['fails'] -= 1
                raise SafetensorError('MetadataIncompleteBuffer')
            return 'payload'

        result = dm.load_cached_tensors(flaky, '/fake/path.safetensors', 'latent')
        assert result == 'payload'
        assert len(no_sleep.calls) == 2, no_sleep.calls
    finally:
        dm.time.sleep = real_sleep


def test_terminal_failure_is_picklable_and_names_file():
    no_sleep = _NoSleep()
    real_sleep = dm.time.sleep
    dm.time.sleep = no_sleep
    try:
        def always_fails():
            raise SafetensorError('HeaderTooLarge')

        try:
            dm.load_cached_tensors(always_fails, '/data/_latent_cache/img_abc.safetensors', 'latent')
            assert False, 'did not raise'
        except RuntimeError as e:
            assert '/data/_latent_cache/img_abc.safetensors' in str(e)
            assert 'HeaderTooLarge' in str(e)
            assert isinstance(e.__cause__, SafetensorError)
            pickle.loads(pickle.dumps(e))  # must survive the worker queue
    finally:
        dm.time.sleep = real_sleep


# ---------------------------------------------------------------------------
# 3. get_latent through the real mixin
# ---------------------------------------------------------------------------

def _make_latent_item(path):
    item = dm.LatentCachingFileItemDTOMixin()
    item.is_latent_cached = True
    item._latent_path = path
    return item


def test_get_latent_corrupt_file_raises_named_runtime_error():
    no_sleep = _NoSleep()
    real_sleep = dm.time.sleep
    dm.time.sleep = no_sleep
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'img_hash.safetensors')
            with open(path, 'wb') as f:
                f.write(b'not a safetensors file')
            item = _make_latent_item(path)
            try:
                item.get_latent()
                assert False, 'did not raise'
            except RuntimeError as e:
                assert path in str(e)
                pickle.loads(pickle.dumps(e))
    finally:
        dm.time.sleep = real_sleep


def test_get_latent_recovers_when_file_repaired_mid_retry():
    # Simulates a transient volume blip: the first read sees garbage, the
    # retry (after "sleeping") sees the real file.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'img_hash.safetensors')
        good = {'latent': torch.randn(4, 8, 8)}

        def repair_on_sleep(seconds):
            save_file(good, path)

        with open(path, 'wb') as f:
            f.write(b'garbage')
        real_sleep = dm.time.sleep
        dm.time.sleep = repair_on_sleep
        try:
            item = _make_latent_item(path)
            latent = item.get_latent()
            assert torch.allclose(latent, good['latent'])
        finally:
            dm.time.sleep = real_sleep


# ---------------------------------------------------------------------------
# 4. load_prompt_embedding through the real mixin
# ---------------------------------------------------------------------------

def _make_text_embed_item(path):
    item = dm.TextEmbeddingFileItemDTOMixin()
    item.is_text_embedding_cached = True
    item._text_embedding_path = path
    return item


def test_load_prompt_embedding_happy_path():
    from toolkit.prompt_utils import PromptEmbeds
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'img_hash.safetensors')
        pe = PromptEmbeds(torch.randn(1, 77, 768))
        pe.save(path)
        item = _make_text_embed_item(path)
        item.load_prompt_embedding()
        assert item.prompt_embeds is not None
        assert torch.allclose(item.prompt_embeds.text_embeds, pe.text_embeds)


def test_load_prompt_embedding_corrupt_file_raises_named_runtime_error():
    no_sleep = _NoSleep()
    real_sleep = dm.time.sleep
    dm.time.sleep = no_sleep
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'img_hash.safetensors')
            with open(path, 'wb') as f:
                f.write(b'definitely not safetensors')
            item = _make_text_embed_item(path)
            try:
                item.load_prompt_embedding()
                assert False, 'did not raise'
            except RuntimeError as e:
                assert path in str(e)
                pickle.loads(pickle.dumps(e))
    finally:
        dm.time.sleep = real_sleep


# ---------------------------------------------------------------------------
# 5. End-to-end: worker exception reaches the main process
# ---------------------------------------------------------------------------

class _ExplodingDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        raise SafetensorError('Error while deserializing header: MetadataIncompleteBuffer')


def test_worker_safetensor_error_reaches_main_process():
    loader = torch.utils.data.DataLoader(
        _ExplodingDataset(),
        batch_size=1,
        num_workers=2,
        # before the unmask fix this hung forever; the timeout turns a
        # regression into a test failure instead of a hang
        timeout=60,
    )
    try:
        next(iter(loader))
        assert False, 'did not raise'
    except SafetensorError as e:
        assert 'MetadataIncompleteBuffer' in str(e)


if __name__ == '__main__':
    tests = [
        ('safetensor_error_module_unmasked', test_safetensor_error_module_unmasked),
        ('safetensor_error_class_and_instance_pickle', test_safetensor_error_class_and_instance_pickle),
        ('exception_wrapper_round_trip', test_exception_wrapper_round_trip),
        ('retry_recovers_from_transient_failure', test_retry_recovers_from_transient_failure),
        ('terminal_failure_is_picklable_and_names_file', test_terminal_failure_is_picklable_and_names_file),
        ('get_latent_corrupt_file_raises_named_runtime_error', test_get_latent_corrupt_file_raises_named_runtime_error),
        ('get_latent_recovers_when_file_repaired_mid_retry', test_get_latent_recovers_when_file_repaired_mid_retry),
        ('load_prompt_embedding_happy_path', test_load_prompt_embedding_happy_path),
        ('load_prompt_embedding_corrupt_file_raises_named_runtime_error', test_load_prompt_embedding_corrupt_file_raises_named_runtime_error),
        ('worker_safetensor_error_reaches_main_process', test_worker_safetensor_error_reaches_main_process),
    ]
    print('cache read resilience tests (issue #49):')
    results = [_run_test(name, fn) for name, fn in tests]
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    sys.exit(1 if failed else 0)
