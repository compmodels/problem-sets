import os
import scipy.io
import requests
import zipfile
from IPython.display import clear_output
from io import BytesIO


def download_similarity_datasets():
    """Downloads human similarity judgment datasets from the URL
    http://www.socsci.uci.edu/~mdlee/all.zip, unzips the file, and
    then returns the path where the files are saved.

    """
    # this is the URL we are downloading
    url = "http://www.socsci.uci.edu/~mdlee/all.zip"

    # download the file and extract its contents.
    request = requests.get(url)
    dest = os.path.join("data", "similarity_data")
    zipfile.ZipFile(BytesIO(request.content)).extractall(dest)

    return dest


def load_dataset(path):
    """Loads the similarity dataset (saved as a .mat file) specified in
    `path`. Returns a dictionary with two keys:

    similarities : array with shape (n, n)
        The similarity judgments, scaled t be between 0 and 1
    names : list of strings, length n
        The names corresponding to the stimuli that were judged,
        such that similarities[i, j] is the similarity of names[i]
        to names[j].

    """
    # load the data in
    data = scipy.io.loadmat(path)
    # pull out the similarity data and the labels for the row/columns
    similarities = 1 - data['d']
    names = list(data['labs'])
    # return them in a dictionary
    return dict(similarities=similarities, names=names)
