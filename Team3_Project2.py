import sys
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def download_kegg(data_dir):
    """
    Download KEGG data files and save them to data dir.

    Input:
    data_dir (str) - path to the directory to save the data files
    
    Files:
    - pathway_names
    - gene_pathway_links
    - gene_list
    
    Returns:
    - Path to the pathway_names file
    - Path to the gene_pathway_links file
    - Path to the gene_list file

    Author: Xiaotong Han
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    pathway_names = data_dir / "pathway_names"
    gene_pathway_links = data_dir / "gene_pathway_links"
    gene_list = data_dir / "gene_list"

    subprocess.run(
        ["curl", "-fL", "-o", str(pathway_names),
         "https://rest.kegg.jp/list/pathway/hsa"],
        check=True
    )
    subprocess.run(
        ["curl", "-fL", "-o", str(gene_pathway_links),
         "https://rest.kegg.jp/link/pathway/hsa"],
        check=True
    )
    subprocess.run(
        ["curl", "-fL", "-o", str(gene_list),
         "https://rest.kegg.jp/list/hsa"],
        check=True
    )

    return pathway_names, gene_pathway_links, gene_list


def read_pathways(pathway_names):
    """
    Read the pathway names file and return a dictionary of 
    pathway IDs and names.
    
    Input:
    pathway_names (str) - path to the pathway names file

    Return:
    pathways (dict) - dictionary pathways[pathway_id] = pathway_name.

    Author: Xiaotong Han
    """
    pathways = {}
    with open(pathway_names, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            assert sum(bool(part.strip()) for part in parts) == 2, \
                f"Unexpected line format: {line}"
            pathway_id, pathway_name = parts
            pathways[pathway_id] = pathway_name
    return pathways


def parse_symbol(text):
    """
    Parse the gene symbol from the last column of the gene list file.
    Input: 
    text (str) - the last column 
    (e.g. MIR6859-1, hsa-mir-6859-1; hsa-miR-6859-3p)
    Return: 
    symbol (str) - the gene symbol (e.g. MIR6859-1)

    Rule of parsing symbol:
    1. If there is a comma, take what's before the comma. 
    - e.g. MIR6859-1, hsa-mir-6859-1; hsa-miR-6859-3p -> MIR6859-1
    2. If there is no comma but semicolon, take what's before the semicolon.
    - e.g. TRN-GTT26-1; tRNA-Asn -> TRN-GTT26-1
    3. If there is no comma or semicolon, take the whole string (full name).
    - e.g. neuroblastoma breakpoint family member 1-like

    Author: Xiaotong Han
    """
    symbol = text.strip().split(',')[0].strip().split(';')[0].strip()
    return symbol


def read_genes(gene_list):
    """
    Input:
    gene_list (str) - path to the gene list file

    Return:
    genes (dict) - dictionary genes[gene_id] = gene_symbol.

    Author: Xiaotong Han
    """
    genes = {}
    with open(gene_list, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            assert sum(bool(part.strip()) for part in parts) == 4, f"Unexpected line format: {line}"
            gene_id, gene_symbol = parts[0], parse_symbol(parts[3])
            genes[gene_id] = gene_symbol
    return genes


def read_links(gene_pathway_links):
    """
    Read the gene-pathway links file and convert the file to a pandas data frame.

    Input:
    gene_pathway_links (str) - path to the gene-pathway links file

    Return:
    A pandas data frame with columns [GENE_ID, PATHWAY_ID]

    The order of rows is as appeared in the gene pathway links file.
    Pathway_ids will start with hsa instead of "path:".

    Author: Xiaotong Han
    """
    links = pd.read_csv(gene_pathway_links, sep='\t', header=None,
                        names=['GENE_ID', 'PATHWAY_ID'])
    links['PATHWAY_ID'] = links['PATHWAY_ID'].str.removeprefix("path:")
    return links


def merge_kegg(links, pathways, genes):
    """
    Merge the links data frame with pathways and genes dictionaries.

    Inputs
    links(pandas dataframe): columns = ['GENE_ID', 'PATHWAY_ID']
    pathways(dictionary): pathways[pathway_id] = pathway_name
    genes(dictionary): genes[gene_id] = gene_symbol

    Return:
    merged (pandas dataframe) - the merged data frame containing gene and pathway information.
    columns: GENE_ID, GENE_SYMBOL, PATHWAY_ID, PATHWAY_NAME
    The order of rows is as appeared in the links data frame.

    Author: Xiaotong Han
    """
    links['GENE_SYMBOL'] = links['GENE_ID'].map(genes)
    links['PATHWAY_NAME'] = links['PATHWAY_ID'].map(pathways)
    return links[['GENE_ID', 'GENE_SYMBOL', 'PATHWAY_ID', 'PATHWAY_NAME']]


def sets_by(merged, key, values):
    """
    This function turns a long table (ie one row per gene pathway pair) into a
    dictionary with one entry per unique value in the key column (part of Task 2).

    Input: the merged table and two column names
    Output: a dictionary, specifically {key: set of values}

    Author: Nina Zhang
    """
    genes_of = {}
    for pathway_id in merged[key].unique():
        genes_of[pathway_id] = set(merged[merged[key] == pathway_id][values])
    return genes_of


def shared_genes(genes_of):
    """
    This function answers which pathways share genes and which genes they share (part of Task 2)

    Input: the dictionary from sets_by (ie {pathway: set of genes})
    Output: a list of (pathway1, pathway2, shared genes) for each pair that overlaps
    by at least one gene. Each pair appears once.

    Author: Nina Zhang
    """
    pairs = []
    for first in genes_of:
        for second in genes_of:
            if first < second:
                shared = genes_of[first] & genes_of[second]
                if len(shared) > 0:
                    pairs.append((first, second, shared))
    return pairs


def crosstalk_table(pairs, names):
    """
    This function builds the table for tasks 2 and 3. KEGG_crosstalk.csv will be saved in main.

    Input: shared_genes output plus the dictionary (pathway_id: name)
    Output: table with PATHWAY_ID1, PATHWAY_NAME1, PATHWAY_ID2, PATHWAY_NAME2,
    NUMBER_OF_OVERLAPPING_GENES, and LIST_OF_OVERLAPPING_GENES, where it is 
    sorted by NUMBER_OF_OVERLAPPING_GENES with the largest number first

    Author: Nina Zhang
    """
    rows = []
    for id1, id2, shared in pairs:
        row = {
            "PATHWAY_ID1": id1,
            "PATHWAY_NAME1": names.get(id1),
            "PATHWAY_ID2": id2,
            "PATHWAY_NAME2": names.get(id2),
            "NUMBER_OF_OVERLAPPING_GENES": len(shared),
            "LIST_OF_OVERLAPPING_GENES": ";".join(sorted(shared)),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.sort_values("NUMBER_OF_OVERLAPPING_GENES", ascending = False)
    return table


def is_overview(pathway_id):
    """
    This function answers whether a pathway is an overview map, which are large 
    KEGG summary maps (part for Task 3).

    Input: A pathway ID string
    Output: True if it falls between hsa01100 and hsa01399 and false otherwise

    Author: Nina Zhang
    """
    number = int(pathway_id.replace("hsa", ""))
    return 1100 <= number <= 1399


def rank_genes(merged):
    """
    Task 4 rank table: GENE_ID, GENE_SYMBOL, NUMBER_OF_PATHWAYS, RANK (largest first).

    Author: David Chang
    """
    # 1. one row per gene-pathway
    pairs = merged[["GENE_ID", "GENE_SYMBOL", "PATHWAY_ID"]].drop_duplicates()

    # 2. count pathways per gene
    counts = (pairs.groupby(["GENE_ID", "GENE_SYMBOL"])
                   .size()
                   .reset_index(name="NUMBER_OF_PATHWAYS"))

    # 3. largest first, ties by symbol
    counts = counts.sort_values(["NUMBER_OF_PATHWAYS", "GENE_SYMBOL"],
                                ascending=[False, True]).reset_index(drop=True)

    # 4. competition rank: 1 + number of genes with strictly more pathways
    ranks = []
    for n in counts["NUMBER_OF_PATHWAYS"]:
        ranks.append(int((counts["NUMBER_OF_PATHWAYS"] > n).sum()) + 1)
    counts["RANK"] = ranks
    return counts


def common_pathways(pathways_of, ranked_genes, n_top=4):
    """
    Return (genes, pathways): the top n_top symbols and the sorted pathway IDs they ALL share.
    Falls back to the top n_top-1, ..., top 2 if there is no common pathway.
    
    Author: David Chang
    """
    for n in range(n_top, 1, -1):
        genes = list(ranked_genes["GENE_SYMBOL"].head(n))
        sets = [pathways_of[g] for g in genes]
        common = set.intersection(*sets)        # same as sets[0] & sets[1] & ...
        if common:
            return genes, sorted(common)
    return genes, []                            # even the top 2 share nothing


def save_common_pathways(genes, pathways, pathway_names, path):
    """
    Write KEGG_top_genes_common_pathways.txt: the genes, then one pathway per line.

    Author: David Chang
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("TOP_GENES\t" + ";".join(genes) + "\n")
        f.write("COMMON_PATHWAYS\n")
        f.writelines([f"{pid}\t{pathway_names.get(pid, '')}\n" for pid in pathways])


def plot_venn3(contents, path):
    """
    Venn diagram of 3 sets. contents = {label: set of items}, exactly 3 entries.

    Author: David Chang
    """
    if len(contents) != 3:
        raise ValueError(f"venn3 needs exactly 3 sets, got {len(contents)}")

    labels = list(contents.keys())
    sets = [set(contents[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    venn3(subsets=sets, set_labels=labels, ax=ax)
    ax.set_title("KEGG pathways of the top 3 genes")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    """
    Author: Xiaotong Han
    """
    # setting up input and output directories
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 Team3_Project2.py DATA_DIR OUT_DIR")

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    # downloading and loading data
    pathway_names, gene_pathway_links, gene_list = download_kegg(data_dir)

    pathways = read_pathways(pathway_names)
    genes = read_genes(gene_list)
    links = read_links(gene_pathway_links)

    # task1: merge
    merged = merge_kegg(links, pathways, genes)

    # task2: shared genes
    dict_genes_of_pathways = sets_by(merged, "PATHWAY_ID", "GENE_SYMBOL")
    list_shared_genes = shared_genes(dict_genes_of_pathways)
    crosstalk_table_all = crosstalk_table(list_shared_genes, pathways)
    crosstalk_table_all.to_csv(out_dir / "KEGG_crosstalk.csv", index=False)

    # task3: shared genes without overview pathways
    merged_no_overview = merged[~merged["PATHWAY_ID"].apply(is_overview)]
    dict_genes_of_pathways_no_overview = sets_by(merged_no_overview, "PATHWAY_ID", "GENE_SYMBOL")
    list_shared_genes_no_overview = shared_genes(dict_genes_of_pathways_no_overview)
    crosstalk_table_no_overview = crosstalk_table(list_shared_genes_no_overview, pathways)
    crosstalk_table_no_overview.to_csv(out_dir / "KEGG_crosstalk_no_overview.csv", index=False)

    # task4: rank genes
    ranked_genes = rank_genes(merged_no_overview)
    ranked_genes.to_csv(out_dir / "KEGG_gene_rank.csv", index=False)

    # task4: top genes and their common pathways
    dict_pathways_of_genes_no_overview = sets_by(merged_no_overview, "GENE_SYMBOL", "PATHWAY_ID")
    list_top_genes, list_common_pathways = common_pathways(
                                            dict_pathways_of_genes_no_overview, ranked_genes
                                            )
    save_common_pathways(list_top_genes, list_common_pathways, pathways,
                         out_dir / "KEGG_top_genes_common_pathways.txt")

    # task5: plot Venn diagram of the top 3 genes
    top3_genes = list(ranked_genes["GENE_SYMBOL"].head(3))
    contents = {gene: dict_pathways_of_genes_no_overview[gene] for gene in top3_genes}
    plot_venn3(contents, out_dir / "KEGG_top3_genes_venn.png")


if __name__ == "__main__":
    main()
