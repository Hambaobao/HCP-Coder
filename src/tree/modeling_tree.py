from typing import List, Dict


class RepoTree():

    def __init__(
        self,
        repo_name_or_path: str,
    ) -> None:

        self.repo_name_or_path = repo_name_or_path
        self.tree = self.build_tree()

    def build_tree(self):
        pass
