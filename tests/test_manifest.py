import pandas as pd
from plagnetplus.data.manifest import validate_manifest

def test_valid_manifest():
    frame = pd.DataFrame({
        "crystal_id": ["x1"],
        "locality": ["aegina"],
        "sample_id": ["Aeg6"],
        "scan_id": ["scan1"],
        "relative_path": ["Aeg6/x1.png"],
        "split": ["all"],
    })
    assert validate_manifest(frame) == []
