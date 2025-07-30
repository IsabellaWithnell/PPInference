
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def load_string_prior(
    tsv_or_taxid: str | int,
    *,
    adata_var_names: Iterable[str],
    score_cutoff: int = 700,
) -> List[Tuple[int, int]]:
    
    if isinstance(tsv_or_taxid, list):
        # Treat as an explicit list of edges like ["GENE1 GENE2", ...] or [(gene1, gene2), ...]
        edges: List[Tuple[int, int]] = []
        name_to_ix = {g: i for i, g in enumerate(adata_var_names)}
        for pair in tsv_or_taxid:
            if isinstance(pair, str):
                parts = pair.strip().split()
                if len(parts) != 2:
                    raise ValueError(f"Expected 'GENEA GENEB' format, got {pair!r}")
                a, b = parts
            elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                a, b = pair
            else:
                raise ValueError(f'Edge specification {pair!r} is neither string nor 2‑tuple.')
            if a in name_to_ix and b in name_to_ix:
                edges.append((name_to_ix[a], name_to_ix[b]))
        return edges
    if isinstance(tsv_or_taxid, int):
        raise NotImplementedError("Automatic STRING download not implemented yet.")

    path = Path(tsv_or_taxid).expanduser()
    df = pd.read_csv(path, sep="\t")
    df = df[df["combined_score"] >= score_cutoff]

    name_to_ix = {g: i for i, g in enumerate(adata_var_names)}
    edges = [
        (name_to_ix[a], name_to_ix[b])
        for a, b in zip(df["protein1"], df["protein2"])
        if a in name_to_ix and b in name_to_ix
    ]
    return edges
