import kagglehub
import shutil
import os

def download_fifa_data(dest="~/2026WorldCupLLM/source_data"):
    dest = os.path.expanduser(dest)
    os.makedirs(dest, exist_ok=True)

    # kagglehub re-downloads only if there's a newer version; otherwise uses cache
    path = kagglehub.dataset_download("mominullptr/fifa-world-cup-2026-dataset")

    for f in os.listdir(path):
        shutil.copy2(os.path.join(path, f), os.path.join(dest, f))

    print("Data available at:", dest)
    return dest

download_fifa_data()