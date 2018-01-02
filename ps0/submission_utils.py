import os
import zipfile
import subprocess


def detect_extra_files():
    """Compare the current problem set against the working tree and return any
    untracked files"""
    extras = subprocess\
        .check_output(["git", "clean", "-dnx"])\
        .decode("utf-8")\
        .replace("\n", "")\
        .split("Would remove ")
    
    # for ps2
    if "data/my_trial_data.npy" in extras:
        extras.remove("data/my_trial_data.npy")
    
    return extras


def zipdir(path, ziph):
    """Recurisvely add files from the current working directory to a zip
    archive, ignoring any untracked files"""
    extras = detect_extra_files()

    if len(extras):
        print("Ignoring:{}\n".format("\n\t".join(extras)))

    print("Creating submission.zip")
    for root, dirs, files in os.walk(path):
        for file in files:
            fp = os.path.join(root, file)

            if fp.replace("./", "") in extras or "ipynb_checkpoints" in fp:
                continue

            print("\tAdding " + fp)
            ziph.write(fp)


def create_submission():
    """Write submission.zip to current working directory and close"""
    zipf = zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('./', zipf)
    zipf.close()