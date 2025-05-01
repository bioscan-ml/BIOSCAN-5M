# BIOSCAN-5M

BIOSCAN-5M Dataset images. 

###### <h3> Image Access
All image packages of the BIOSCAN-5M dataset are available in the [GoogleDrive](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0),
within [BIOSCAN_5M_IMAGES](https://drive.google.com/drive/u/1/folders/1tZ5V_qWSPdDwD90oLz_Uqykp1AoBzLVM) folder.
Accessing the dataset images is facilitated by the following directory structure used to organize the dataset images:

```plaintext
bioscan5m/images/[imgtype]/[split]/[chunk]/[{processid}.jpg]
```

### BIOSCAN-5M Image Data Structure

The BIOSCAN-5M image data is organized within the `BIOSCAN_5M_original_256.zip` and `BIOSCAN_5M_cropped_256.zip` packages as follows:

- **`[imgtype]`**: Type of the image, which can be one of the following:
  - `original_full`
  - `cropped`
  - `cropped_256`
  - `original_256`

- **`[split]`**: Data split, which can be one of the following:
  - `pretrain`
  - `train`
  - `val`
  - `test`
  - `val_unseen`
  - `test_unseen`
  - `key_unseen`
  - `other_heldout`

- **`[chunk]`**: Determined by using the first two or one characters of the MD5 checksum (in hexadecimal) of the `[processid]`:
  - **`pretrain` split**: Files are organized into 256 directories, using the first two characters of the MD5 checksum of the `[processid]`.
  - **`train` and `other_heldout` splits**: Files are organized into 16 directories, using the first letter of the MD5 checksum of the `[processid]`.
  - **`val`, `test`, `val_unseen`, `test_unseen`, `key_unseen`, `other_heldout` splits**: These splits do not use chunk directories, as they contain fewer than 50k images each.

- **`[processid]`**: A unique identifier assigned by the International Barcode of Life Consortium (BOLD).


### Zip File Structure

- **`BIOSCAN_5M_original_256.zip`** and **`BIOSCAN_5M_cropped_256.zip`** packages are also available as separate zip files for each data split. This facilitates experiments and enables separate downloading of specific splits as needed for different experiments:
  - `pretrain`, `train`, and `eval` splits.
  - The evaluation splits: `val`, `test`, `val_unseen`, `test_unseen`, `key_unseen`, and `other_heldout`, are all part of the evaluation partition of both the `original_256` and `cropped_256` image packages.



