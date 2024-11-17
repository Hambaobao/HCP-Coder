import os

ROOT = "benchmarks/crosscodeeval/raw_data"
TEMP = "benchmarks/crosscodeeval/real_repos"


def build_repo(item: dict):
    final_repo_path = os.path.join(TEMP, item["task_id"], item["repository"])
    if os.path.exists(final_repo_path):
        return final_repo_path

    task_id = item["task_id"]
    repo_id = item["repository"]
    file = item["file"]
    hole_row_idx = item["hole_row_idx"]
    hole_col_idx = item["hole_col_idx"]

    repo_path = os.path.join(ROOT, repo_id)
    temp_repo_path = os.path.join(TEMP, task_id)
    os.makedirs(temp_repo_path, exist_ok=True)
    os.system(f"cp -r {repo_path} {temp_repo_path}")

    file_path = os.path.join(final_repo_path, file)
    with open(file_path, "r") as f:
        lines = f.readlines()
    hole_line = lines[hole_row_idx - 1][:hole_col_idx]

    prefix = "".join(lines[:hole_row_idx - 1])
    suffix = "\n" + "".join(lines[hole_row_idx:])
    content = prefix + hole_line + suffix
    with open(file_path, "w") as f:
        f.write(content)

    return final_repo_path
