import kagglehub
import os
import shutil


def main():
    path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
    dest = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(path):
        shutil.copy2(os.path.join(path, file), os.path.join(dest, file))
    print("Downloaded files to:", dest)


if __name__ == "__main__":
    main()
