import os
import shutil
import subprocess

# List of GitHub repository URLs
repo_urls = [
    "https://github.com/geekcomputers/Python.git" ,
    "https://github.com/OmkarPathak/Python-Programs.git" ,
    "https://github.com/Asabeneh/30-Days-Of-Python.git" ,
    "https://github.com/avinashkranjan/Amazing-Python-Scripts.git",
    "https://github.com/TheAlgorithms/Python.git",
    "https://github.com/CodeWithHarry/The-Ultimate-Python-Course.git",
    "https://github.com/arnab132/Graph-Plotting-Python.git",
    "https://github.com/matplotlib/matplotlib.git",
    "https://github.com/donnemartin/data-science-ipython-notebooks.git",
    "https://github.com/kanchanchy/Data-Visualization-in-Python.git",
    "https://github.com/madhurimarawat/Data-Visualization-using-python.git",
    "https://github.com/rojaAchary/Data-Visualization-with-Python.git"
]


destination_folder = "Auto-Code-Python\RawFiles"


os.makedirs(destination_folder, exist_ok=True)


for repo_url in repo_urls:
    repo_name = repo_url.split("/")[-1].replace(".git", "")  # Extract repo name
    repo_path = os.path.join(destination_folder, repo_name)

    if os.path.exists(repo_path):
        print(f"Repository {repo_name} already exists. Skipping...")
    else:
        print(f"Cloning {repo_name} into {repo_path}...")
        subprocess.run(["git", "clone", repo_url, repo_path])

print("All repositories have been cloned successfully!")

def move_python_files(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(source_folder, topdown=False):
        for file in files:
            if file.endswith(".py") or file.endswith(".ipynb"):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(destination_folder, file)

                # Ensure filename uniqueness
                counter = 1
                while os.path.exists(dst_file):
                    name, ext = os.path.splitext(file)
                    dst_file = os.path.join(destination_folder, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(src_file, dst_file)  # Move file

    print("All .py files moved successfully!")


data_folder = "Dataset"
move_python_files(destination_folder, data_folder)
