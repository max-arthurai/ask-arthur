import os
import pandas as pd


def process_csv(file_path):
    df_full = pd.read_csv(file_path)
    for idx, row in df_full.iterrows():
        print(idx)
        ctype = row['content_type']
        src = row['source']
        # file_name = ctype + src.replace("https://", "").replace("www.", "").replace("docs.", "").replace("arthur.ai/", "").replace("blog/", "").replace("bench.readthedocs.io/en/latest/", "").replace("docs/", "").replace("edit/", "").replace("sdk.", "").replace("staging.", "").replace("legacy.", "").replace("/", "-").replace("html", "") + ".txt"
        if not os.path.exists(f"docs/txt_files/{ctype}.txt"):
            with open(f"docs/txt_files/{ctype}.txt", "w") as f:
                f.write("")
        with open(f"docs/txt_files/{ctype}.txt", 'a', encoding='utf-8') as file:
            if pd.isna(row['text']):
                file.write("")
            else:
                file.write(f"Content type: {ctype}\nSource: {src}\n {row['text']}\n=====================\n")
        print(f"Extended {ctype} doc")


if __name__ == "__main__":
    path_to_csv = "arthur_index_315.csv"  # Change this to the path of your CSV file
    process_csv(path_to_csv)
