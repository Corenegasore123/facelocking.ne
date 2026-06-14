# src/rebuild_db.py
"""
Rebuild face database from existing enrollment crops.

This script scans data/enroll/ for all person folders, loads their aligned crops,
re-embeds them, and rebuilds data/db/face_db.npz and face_db.json.

Use this if:
- You have crops but the database is missing/outdated
- You want to rebuild after model changes
- Multiple people were enrolled but only some are in the DB

Run:
python -m src.rebuild_db
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from .embed import ArcFaceEmbedderONNX
from .enroll import EnrollConfig, ensure_dirs, save_db, mean_embedding, _list_existing_crops


def rebuild_database_from_crops(
    cfg: EnrollConfig,
    emb: ArcFaceEmbedderONNX,
) -> Dict[str, np.ndarray]:
    """
    Scan crops_dir for person folders, load all crops, compute mean embeddings.
    Returns: {name: embedding_vector}
    """
    db: Dict[str, np.ndarray] = {}
    
    if not cfg.crops_dir.exists():
        print(f"Enrollment directory {cfg.crops_dir} does not exist.")
        return db
    
    # Find all person folders
    person_dirs = [d for d in cfg.crops_dir.iterdir() if d.is_dir()]
    
    if not person_dirs:
        print(f"No person folders found in {cfg.crops_dir}")
        return db
    
    print(f"Found {len(person_dirs)} person folder(s):")
    for pd in person_dirs:
        print(f"  - {pd.name}")
    
    print("\nProcessing crops and computing embeddings...")
    
    for person_dir in sorted(person_dirs):
        name = person_dir.name
        crops = _list_existing_crops(person_dir, cfg.max_existing_crops)
        
        if not crops:
            print(f"  {name}: No crops found, skipping.")
            continue
        
        print(f"  {name}: Loading {len(crops)} crops...", end=" ", flush=True)
        embeddings: List[np.ndarray] = []
        
        for crop_path in crops:
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            try:
                r = emb.embed(img)
                embeddings.append(r.embedding)
            except Exception as e:
                print(f"\n    Warning: Failed to embed {crop_path.name}: {e}")
                continue
        
        if not embeddings:
            print(f"Failed - no valid embeddings.")
            continue
        
        template = mean_embedding(embeddings)
        db[name] = template
        print(f"OK - computed mean embedding from {len(embeddings)} samples.")
    
    return db


def main():
    cfg = EnrollConfig()
    ensure_dirs(cfg)
    
    print("=" * 60)
    print("Rebuilding Face Database from Enrollment Crops")
    print("=" * 60)
    print(f"Enrollment directory: {cfg.crops_dir}")
    print(f"Output database: {cfg.out_db_npz}")
    print()
    
    # Initialize embedder
    emb = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    
    # Rebuild database
    db = rebuild_database_from_crops(cfg, emb)
    
    if not db:
        print("\nNo identities found. Database not updated.")
        return
    
    # Save database
    print(f"\nSaving database with {len(db)} identities...")
    
    # Compute metadata
    sample_counts: Dict[str, int] = {}
    for name in db.keys():
        person_dir = cfg.crops_dir / name
        crops = _list_existing_crops(person_dir, cfg.max_existing_crops)
        sample_counts[name] = len(crops)
    
    total_samples = sum(sample_counts.values())
    first_emb = next(iter(db.values()))
    
    meta = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": int(first_emb.size),
        "names": sorted(db.keys()),
        "samples_existing_used": total_samples,
        "samples_new_used": 0,
        "samples_total_used": total_samples,
        "samples_per_person": sample_counts,
        "note": "Embeddings are L2-normalized vectors. Matching uses cosine similarity.",
    }
    
    save_db(cfg, db, meta)
    
    print("\n" + "=" * 60)
    print("Database rebuilt successfully!")
    print("=" * 60)
    print(f"Identities: {', '.join(sorted(db.keys()))}")
    print(f"Total samples: {total_samples}")
    print(f"Database saved to: {cfg.out_db_npz}")
    print(f"Metadata saved to: {cfg.out_db_json}")
    print()


if __name__ == "__main__":
    main()
