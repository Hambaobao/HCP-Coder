from src.topo import RepoTopo
from src.retriever import AutoRetriever

import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='RepoTopo')

    parser.add_argument('-d', '--repo', type=str, required=True, help='Directory to the repository')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the current file')
    parser.add_argument('-r', '--row', type=int, required=True, help='Row number of the fim hole in current file')
    parser.add_argument('-c', '--col', type=int, required=True, help='Column number of the fim hole in current file')
    parser.add_argument('-k', '--top-k', type=int, default=[5], help='Number of top k results to retrieve')
    parser.add_argument('-p', '--top-p', type=float, default=[0.3], help='Top p value for retrieval')

    return parser.parse_args()


def main():

    # Parse the arguments
    args = parse_args()

    # Initialize the RepoTopo object
    repo_topo = RepoTopo(args.path)

    # define the retriever used to retrieve the related functions
    # we also support engine: 'jina'
    retriever = AutoRetriever(engine='openai')

    # get the file node object of the current file
    file_node = repo_topo.file_nodes[args.file]

    # get the hierarchical cross file context
    cross_file_context = repo_topo.get_hierarchical_cross_file_context(
        retriever,
        file_node,
        row=args.row,
        col=args.col,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # we use dense infile context for all strategies
    infile_context = repo_topo.get_infile_context(file_node, args.row, args.col)

    print('Cross File Context:', cross_file_context)

    print('Infile Context:', infile_context)

    # -----------------------------------------------
    # You can use these information to implement your own application logic
    # -----------------------------------------------
