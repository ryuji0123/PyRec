import os
import zipfile

from collections import namedtuple
from six.moves.urllib.request import urlretrieve

def get_dataset_dir():
    folder = os.environ.get("PYREC_DATASET_FOLDER", os.path.join(os.path.expanduser("~"), ".pyrec_dataset"))
    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder

BuiltinDataset = namedtuple("DefaultDataset", ["url", "path", "reader_params"])

BUILTIN_DATASETS = {
    "ml-100k":
        BuiltinDataset(
            url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
            path=os.path.join(get_dataset_dir(), "ml-100k", "ml-100k", "u.data"),
            reader_params=dict(
                line_format="user item rating timestamp",
                rating_scale=(1, 5),
                sep="\t"
            )
        ),
    "ml-1m":
        BuiltinDataset(
            url="http://files.grouplens.org/datasets/movielens/ml-1m.zip",
            path=os.path.join(get_dataset_dir(), "ml-1m", "ml-1m", "ratings.dat"),
            reader_params=dict(
                line_format="user item rating timestamp",
                rating_sclae=(1, 5),
                sep="::"
            )
        ),
}

def download_builtin_dataset(name):
    dataset = BUILTIN_DATASETS[name]

    print(f"Trying to download dataset from {dataset.url} ...")
    tmp_file_path = os.path.join(get_dataset_dir(), "tmp.zip")
    urlretrieve(dataset.url, tmp_file_path)

    with zipfile.ZipFile(tmp_file_path, "r") as tmp_zip:
        tmp_zip.extractall(os.path.join(get_dataset_dir(), name))

    os.remove(tmp_file_path)
    print(f"Done! Dataset {name} has been saved to {os.path.join(get_dataset_dir(), name)}")