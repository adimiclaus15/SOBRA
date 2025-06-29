import argparse
import pathlib

import numpy as np
import pydicom


def process_dicom(dcm_path: pathlib.Path, out_base: pathlib.Path, in_base: pathlib.Path):
    ds = pydicom.dcmread(str(dcm_path), force=True)
    raw_pixels = ds.pixel_array.astype(float)

    
    flat = raw_pixels.flatten()
    positive = flat[flat > 0]
    rounded = np.rint(positive).astype(int)

    
    rel = dcm_path.relative_to(in_base)
    rel_parent = rel.parent
    out_dir = out_base / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    
    out_filename = dcm_path.stem + "_positive.txt"
    out_filepath = out_dir / out_filename

    
    with open(out_filepath, "w") as f:
        f.write(f"{len(rounded)}\n")
        for val in rounded:
            f.write(f"{val}\n")

    print(f"Processed: {dcm_path} → {out_filepath} (count = {len(rounded)})")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively traverse an input directory of DICOM files. "
            "For each .dcm file, extract pixel_array as floats, keep only values > 0, "
            "round them to integers, and write a single text file whose first line "
            "is the number of positive values, followed by one integer per line. "
            "The output folder structure mirrors the input."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Path to the root directory containing subfolders of DICOM (.dcm) files.",
    )
    parser.add_argument(
        "--outdir",
        default="positive_values_output",
        help="Directory where output folders and text files will be written "
             "(default: “positive_values_output”).",
    )

    args = parser.parse_args()
    in_base = pathlib.Path(args.input_dir)
    if not in_base.exists() or not in_base.is_dir():
        raise FileNotFoundError(f"Cannot find input directory at: {in_base}")

    out_base = pathlib.Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)


    dicom_files = list(in_base.rglob("*.dcm"))
    if not dicom_files:
        print(f"No DICOM files (*.dcm) found under {in_base}.")
        return

    for dcm_path in dicom_files:
        try:
            process_dicom(dcm_path, out_base, in_base)
        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")


if __name__ == "__main__":
    main()

