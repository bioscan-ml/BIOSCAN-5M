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

| Attributes  | Categories | Total Count U.B | Mean U.B| Median U.B| Std. Dev. U.B  | Avg SDI U.B | **Mean** P.D. | **Std. Dev.** P.D |
|-------------|------------|-----------------|---------|-----------|----------------|-------------|---------------|-------------------|
| `phylum`    | 1          | 2,486,492       |         |           |                | 19.78       | 158           | 42                |
| `class`     | 10         | 2,482,891       | 248,289 | 177       | 725,237        | 8.56        | 166           | 103               |
| `order`     | 55         | 2,474,855       | 44,997  | 57        | 225,098        | 7.05        | 128           | 53                |
| `family`    | 934        | 2,321,301       | 2,485   | 46        | 19,701         | 5.42        | 90            | 46                |
| `subfamily` | 1,542      | 657,639         | 426     | 17        | 3,726          | 4.28        | 78            | 51                |
| `genus`     | 7,605      | 531,109         | 70      | 5         | 1,061          | 2.63        | 50            | 39                |
| `species`   | 22,622     | 202,260         | 9       | 2         | 37             | 1.46        | 17            | 18                |
| `BIN`       | 324,411    | 2,474,000       | 9       | 2         | 37             | 1.29        | N/A           | N/A               |

**U.B:** Unique DNA Barcode Sequence  
**P.D.** Pairwise Distance 

```bash
python main.py --download --file_to_download <file_name> --ID_mapping_path <path/bioscan5m_dataset_file_ID_mapping.txt>
``` 

The list of files available for download from GoogleDrive are:

 - Metadata: BIOSCAN_5M_Insect_Dataset_metadata_MultiTypes.zip
 - Bounding Box: BIOSCAN_5M_Insect_bbox.tsv
 - BIOSCAN_5M_original_full.{01:05}.zip
 - BIOSCAN_5M_original_256.zip
    - BIOSCAN_5M_original_256_pretrain.zip
    - BIOSCAN_5M_original_256_train.zip
    - BIOSCAN_5M_original_256_eval.zip
 - BIOSCAN_5M_cropped.{01:02}.zip
 - BIOSCAN_5M_cropped_256.zip
    - BIOSCAN_5M_cropped_256_pretrain.zip
    - BIOSCAN_5M_cropped_256_train.zip
    - BIOSCAN_5M_cropped_256_eval.zip

###### <h3> Dataset Statistics
To see the statistics of the BIOSCAN-5M dataset, run the following:

```bash
python main.py --attr_stat <attribute_type>
``` 
The attribute type can be <code>genetic</code>, <code>geographic</code>, and <code>size</code>.

### <h4> Statistics of BIOSCAN-5M dataset records by taxonomic ranks, DNA sequence, and BIN.
| Attributes        | Imbalance Ratio (IR) | Categories  | Labelled        | Labelled (%)    |
|-------------------|----------------------|-------------|-----------------|-----------------|
| **phylum**        | 1                    | 1           | 5,150,850       | 100.0           |
| **class**         | 719,831              | 10          | 5,146,837       | 99.9            |
| **order**         | 3,675,317            | 55          | 5,134,987       | 99.7            |
| **family**        | 938,928              | 934         | 4,932,774       | 95.8            |
| **subfamily**     | 323,146              | 1,542       | 1,472,548       | 28.6            |
| **genus**         | 200,268              | 7,605       | 1,226,765       | 23.8            |
| **species**       | 7,694                | 22,622      | 473,094         | 9.2             |
| **dna_bin**       | 35,458               | 324,411     | 5,137,441       | 99.7            |
| **dna_barcode**   | 3,743                | 2,486,492   | 5,150,850       | 100.0           |


To see the statistics of a BIOSCAN-5M dataset attribute, run the following:

```bash
python main.py --attr <attribute>
``` 

The attribute can be one of the genetic (e.g., order), geographic (e.g., country) or size (e.g., scale_factor) attributes.

###### <h3> Dataset Distribution
To see the category distribution of the BIOSCAN-5M dataset, run the following:

```bash
python main.py --attr_dist <attribute_type>
``` 
The attribute type can be <code>genetic</code>, <code>geographic</code>, and <code>size</code>.

###### <h3> Taxonomic Class Distribution
<div align="center">
  <img src="https://github.com/zahrag/BIOSCAN-5M/blob/main/BIOSCAN_images/repo_images/class_order_stats.png" 
       alt="An array of category distribution of the taxonomic level class." />
  <p><b>Figure 2:</b> The category distribution of the taxonomic level class of the BIOSCAN-5M dataset.</p>
</div>

###### <h3> Insect Statistics
To see the statistics of the class Insect of the BIOSCAN-5M dataset, run the following:

```bash
python main.py --level_name class Insecta --attr_stat <attribute_type>
``` 
The attribute type can be <code>genetic</code>, <code>geographic</code>, and <code>size</code>.

###### <h3> Non-insect organisms
In addition to insects (98% of specimens), the BIOSCAN-5M dataset also contains arthropods from non-insect taxonomic classes.
These are primarily arachnids and springtails (Collembola).

<div align="center">
  <img src="https://github.com/zahrag/BIOSCAN-5M/blob/main/BIOSCAN_images/repo_images/non_insect.png" 
       alt="An array of example non-insect arthropod images from the BIOSCAN-5M dataset." />
  <p><b>Figure 7:</b> Examples of the original images of non-insect organisms.
</div>


###### <h3> Challenges and Limitations
The BIOSCAN-5M dataset faces some challenges and limitations:
- Sampling bias
  - It exposes a sampling bias as a result of the geographical locations where and the methods through which organisms were collected.
- Accessing ground-truth labels
  - The number of labelled records sharply declines as we delve deeper into taxonomic ranking groups, particularly when moving towards finer-grained taxonomic ranks beyond the family level.
- Fine-grained classification
  - The class imbalance ratio (IR) of each taxonomic group highlights a notable disparity in sample numbers between the majority class (the class with the most samples) and the minority class(es) (those with fewer samples)
  - The high similarity between images of distinct categories.

<div align="center">
  <img src="https://github.com/zahrag/BIOSCAN-5M/blob/main/BIOSCAN_images/repo_images/species_f.png" 
       alt="Sample images of distinct species from the Order Diptera." />
  <p><b>Figure 2:</b> Sample images of distinct species from the Order Diptera, 
                      which comprises about 50% of our dataset. High similarity between samples of different species 
highlights significant classification challenges.</p>
</div>


###### <h3> Dataset Split
You can access our proposed data splitting approach using the Python script `bioscan_split.py.
`
###### <h4> Statistics and Purpose of Our Data Partitions

| Species Set | Split              | Purpose                        | No. of Samples | No. of Barcodes | No. of Species |
|-------------|--------------------|--------------------------------|----------------|-----------------|----------------|
| Unknown     | `pretrain`         | SSL, semi-sup. training        | 4,677,756      | 2,284,232       | ---            |
| Seen        | `train`            | Supervision; retrieval keys    | 289,203        | 118,051         | 11,846         |
|             | `val`              | Model dev; retrieval queries   | 14,757         | 6,588           | 3,378          |
|             | `test`             | Final eval; retrieval queries  | 39,373         | 18,362          | 3,483          |
| Unseen      | `key_unseen`       | Retrieval keys                 | 36,465         | 12,166          | 914            |
|             | `val_unseen`       | Model dev; retrieval queries   | 8,819          | 2,442           | 903            |
|             | `test_unseen`      | Final eval; retrieval queries  | 7,887          | 3,401           | 880            |
| Heldout     | `other_heldout`    | Novelty detector training      | 76,590         | 41,250          | 9,862          |