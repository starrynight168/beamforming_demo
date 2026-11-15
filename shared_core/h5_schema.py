"""HDF5 schema version helpers."""

import h5py


H5_SCHEMA_VERSION = 3


def write_h5_schema(hf):
    hf.create_dataset("h5_schema_version", data=H5_SCHEMA_VERSION, dtype="u1")


def require_h5_schema(hf):
    if "h5_schema_version" not in hf:
        raise ValueError("H5 缺少 h5_schema_version；旧 H5 必须重新打包")
    dataset = hf["h5_schema_version"]
    if dataset.shape != () or dataset.dtype.kind not in "iu":
        raise ValueError("H5 h5_schema_version must be a scalar integer")
    version = int(dataset[()])
    if version != H5_SCHEMA_VERSION:
        raise ValueError(
            f"不支持 H5 schema {version}，当前要求 {H5_SCHEMA_VERSION}；请重新打包"
        )


def validate_h5_schema(path):
    with h5py.File(path, "r") as hf:
        require_h5_schema(hf)
