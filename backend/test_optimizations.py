"""Quick smoke test for the two temporal search optimizations."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# --- Test 1: Numba kernels ---
print("=== Test 1: Numba JIT kernels ===")
try:
    from src.ui.temporal_dp_numba import (
        run_dp_on_window,
        has_near_tied_suffix_choices,
        window_temporal_candidates,
        suffix_temporal_candidates,
        select_non_overlapping,
    )
    
    np.random.seed(42)
    S = np.random.randn(3, 10).astype(np.float32)
    ts = np.linspace(0, 9, 10).astype(np.float32)
    
    # DP on window
    score, path, plen = run_dp_on_window(S)
    print(f"  run_dp_on_window: score={score:.4f}, path_len={plen}")
    
    # Near-tied check
    tied = has_near_tied_suffix_choices(S)
    print(f"  has_near_tied: {tied}")
    
    # Window candidates
    scores, pf, po, st, et, pl = window_temporal_candidates(S, ts, -1.0)
    print(f"  window_candidates: {len(scores)} candidates")
    
    # Suffix candidates
    scores2, pf2, po2, st2, et2, pl2 = suffix_temporal_candidates(S, ts)
    print(f"  suffix_candidates: {len(scores2)} candidates")
    
    # NMS
    if len(scores) > 0:
        sel = select_non_overlapping(scores, pf, po, st, et, int(pl), 3, 0.6)
        print(f"  select_non_overlapping: {len(sel)} selected")
    
    print("  [OK] All Numba kernels work!\n")
except Exception as e:
    print(f"  [FAIL] {e}\n")

# --- Test 2: EmbeddingMemmapStore ---
print("=== Test 2: EmbeddingMemmapStore ===")
try:
    import tempfile
    from src.ui.embedding_memmap import EmbeddingMemmapStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some fake .npy files
        emb_dir = os.path.join(tmpdir, "embs")
        os.makedirs(emb_dir)
        
        paths = []
        dim = 512
        for i in range(50):
            p = os.path.join(emb_dir, f"frame_{i:04d}.npy")
            np.save(p, np.random.randn(dim).astype(np.float32))
            paths.append(p)
        
        # Build
        store_dir = os.path.join(tmpdir, "store")
        store = EmbeddingMemmapStore.build(paths, store_dir)
        print(f"  Built store: shape={store.shape}, len={len(store)}")
        
        # Reload
        store2 = EmbeddingMemmapStore.load(store_dir)
        print(f"  Loaded store: shape={store2.shape}")
        
        # Lookup
        emb = store2[paths[0]]
        print(f"  Single lookup: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
        
        # Batch lookup
        batch = store2.get_batch(paths[:5])
        print(f"  Batch lookup: shape={batch.shape}")
        
        # Containment check
        assert paths[0] in store2
        assert "/nonexistent.npy" not in store2
        
        # Release memmap references before temp dir cleanup (Windows)
        del store, store2
        
    print("  [OK] EmbeddingMemmapStore works!\n")
except Exception as e:
    print(f"  [FAIL] {e}\n")

# --- Test 3: Integration in temporal_search ---
print("=== Test 3: temporal_search imports ===")
try:
    from src.ui.temporal_search import (
        _temporal_topk_dp,
        set_memmap_store,
        get_memmap_store,
        _HAS_NUMBA,
    )
    print(f"  _HAS_NUMBA = {_HAS_NUMBA}")
    print("  [OK] temporal_search imports work!\n")
except Exception as e:
    print(f"  [FAIL] {e}\n")

print("All tests complete.")
