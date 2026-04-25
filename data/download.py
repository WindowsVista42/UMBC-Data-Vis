import ast
import kagglehub
import os
import shutil

import pandas as pd

ARRAY_COLUMNS = ["tags", "nutrition", "steps", "ingredients"]


def convert_recipes_to_jsonl(dest):
    df = pd.read_csv(os.path.join(dest, "RAW_recipes.csv"))
    for col in ARRAY_COLUMNS:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    out = os.path.join(dest, "RAW_recipes.jsonl")
    df.to_json(out, orient="records", lines=True)
    print("Converted RAW_recipes.csv to:", out)


def main():
    path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
    dest = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(path):
        shutil.copy2(os.path.join(path, file), os.path.join(dest, file))
    print("Downloaded files to:", dest)
    convert_recipes_to_jsonl(dest)


if __name__ == "__main__":
    main()
