import os
import subprocess
import sys

output_dir = "/home/jovyan/shares/SR006.nfs2/vashina/datasets/MeViS/data_sets/valid"
os.makedirs(output_dir, exist_ok=True)

# gdown downloads Google Drive files/folders correctly (wget only gets the HTML page)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        "https://drive.google.com/drive/folders/1aLUazTvMqpgg6SWz_gHJOvMbrItvHi9t",
        "-O",
        output_dir,
        "--remaining-ok",
    ],
    check=True,
)