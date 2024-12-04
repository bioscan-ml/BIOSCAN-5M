# BIOSCAN-5M

Here, you will find an overview of the statistical analysis performed on the BIOSCAN-5M DNA nucleotide barcode sequences. 
For more details, please refer to the [BIOSCAN-5M paper](https://arxiv.org/abs/2406.12723).


## Statistical Analysis of the identical DNA Barcodes

The DNA barcode statistics for various taxonomic ranks in the BIOSCAN-5M dataset. 
We indicate the total number of unique barcodes for the samples labelled to a given rank, and the mean, median, and standard deviation of the number of unique barcodes within the subgroupings at that rank. We also show the average across subgroups of the Shannon Diversity Index (SDI) for the DNA barcodes, measured in bits. We report the mean and standard deviation of pairwise DNA barcode sequence distances, aggregated across subgroups for each taxonomic rank.


The following table presents the DNA barcode statistics for various taxonomic ranks in the BIOSCAN-5M dataset. The table includes:

- **Total Count** number of unique barcodes for the samples labeled at each rank.
- **Mean**, **Median**, and **Standard Deviation** of the number of unique barcodes within the subgroupings at that rank.
- **Average Shannon Diversity Index (SDI)** for the DNA barcodes (measured in bits).
- **Mean** and **Standard Deviation** of pairwise DNA barcode sequence distances, aggregated across subgroups for each taxonomic rank.

| **Level**   | **Categories** | **Total Count** U.B | **Mean** U.B | **Median** U.B | **Std. Dev.** U.B | **Avg SDI** U.B | **Mean** P.D. | **Std. Dev.** P.D |
|-------------|----------------|---------------------|--------------|----------------|-------------------|-----------------|---------------|-------------------|
| `phylum`    | 1              | 2,486,492           |              |                |                   | 19.78           | 158           | 42                |
| `class`     | 10             | 2,482,891           | 248,289      | 177            | 725,237           | 8.56            | 166           | 103               |
| `order`     | 55             | 2,474,855           | 44,997       | 57             | 225,098           | 7.05            | 128           | 53                |
| `family`    | 934            | 2,321,301           | 2,485        | 46             | 19,701            | 5.42            | 90            | 46                |
| `subfamily` | 1,542          | 657,639             | 426          | 17             | 3,726             | 4.28            | 78            | 51                |
| `genus`     | 7,605          | 531,109             | 70           | 5              | 1,061             | 2.63            | 50            | 39                |
| `species`   | 22,622         | 202,260             | 9            | 2              | 37                | 1.46            | 17            | 18                |
| `BIN`       | 324,411        | 2,474,881           | 8            | 2              | 40                | 1.29            | N/A           | N/A               |

**U.B:** Unique DNA Barcode Sequence  
**P.D.** Pairwise Distance 

## Pairwise Damerau-Levenshtein Distance Analysis of identical DNA barcodes

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/zahrag/BIOSCAN-5M/main/BIOSCAN_images/repo_images/class_distance_distribution.png" 
         alt="class." />
    <figcaption><b>Figure 1: </b>Distribution of pairwise distances of subgroups of class. The x-axis shows the subgroup
    categories sorted alphabetically.</figcaption>
  </figure>
</div>

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/zahrag/BIOSCAN-5M/main/BIOSCAN_images/repo_images/order_distance_distributions.png" 
         alt="order." />
    <figcaption><b>Figure 2: </b>Distribution of pairwise distances of subgroups of order. The x-axis shows the subgroup
    categories sorted alphabetically.</figcaption>
  </figure>
</div>

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/zahrag/BIOSCAN-5M/main/BIOSCAN_images/repo_images/species_distance_distribution.png" 
         alt="species." />
  </figure>
</div>
<div align="left">
  <p><b>Figure 3:</b> Distribution of pairwise distances of subgroups of species. Among the species, there are
    8,372 distinct subgroups with sufficient identical barcodes for calculating pairwise distances, which
    makes visualization challenging. To address this, the groups are sorted in descending order based
    on their mean distances and partitioned into 100 bins. These bins are used to plot the distribution
    of pairwise distances within the species rank. The mean distance of each bin is displayed along the
    x-axis.</div>
