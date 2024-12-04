# BIOSCAN-5M

Here you will get some clue about BIOSCAN-5M statictical analysis of the DNA nucleotide barcode sequences:


###### <h3> Dataset Download




### DNA Barcode Statistics

The DNA barcode statistics for various taxonomic ranks in the BIOSCAN-5M dataset. We indicate the total number of unique barcodes for the samples labelled to a given rank, and the mean, median, and standard deviation of the number of unique barcodes within the subgroupings at that rank. We also show the average across subgroups of the Shannon Diversity Index (SDI) for the DNA barcodes, measured in bits. We report the mean and standard deviation of pairwise DNA barcode sequence distances, aggregated across subgroups for each taxonomic rank.

## DNA Barcode Statistics

The following table presents the DNA barcode statistics for various taxonomic ranks in the BIOSCAN-5M dataset. The table includes:

- **Total Count** number of unique barcodes for the samples labeled at each rank.
- **Mean**, **Median**, and **Standard Deviation** of the number of unique barcodes within the subgroupings at that rank.
- **Average Shannon Diversity Index (SDI)** for the DNA barcodes (measured in bits).
- **Mean** and **Standard Deviation** of pairwise DNA barcode sequence distances, aggregated across subgroups for each taxonomic rank.

| Attributes  | Categories | Total Count U.B | Mean U.B | Median U.B| Std. Dev. U.B | Avg SDI U.B | **Mean** P.D. | **Std. Dev.** P.D |
|-------------|------------|-----------------|----------|-----------|---------------|-------------|---------------|-------------------|
| `phylum`    | 1          | 2,486,492       |          |           |               | 19.78       | 158           | 42                |
| `class`     | 10         | 2,482,891       | 248,289  | 177       | 725,237       | 8.56        | 166           | 103               |
| `order`     | 55         | 2,474,855       | 44,997   | 57        | 225,098       | 7.05        | 128           | 53                |
| `family`    | 934        | 2,321,301       | 2,485    | 46        | 19,701        | 5.42        | 90            | 46                |
| `subfamily` | 1,542      | 657,639         | 426      | 17        | 3,726         | 4.28        | 78            | 51                |
| `genus`     | 7,605      | 531,109         | 70       | 5         | 1,061         | 2.63        | 50            | 39                |
| `species`   | 22,622     | 202,260         | 9        | 2         | 37            | 1.46        | 17            | 18                |
| `BIN`       | 324,411    | 2,474,881       | 8        | 2         | 40            | 1.29        | N/A           | N/A               |

**U.B:** Unique DNA Barcode Sequence  
**P.D.** Pairwise Distance 

