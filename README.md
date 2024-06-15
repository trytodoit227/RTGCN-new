# Representation Learning of Temporal Graphs with Structural Roles
## 1. Environment

`pip install -r requirements.txt`

## 2. File Structure
- `code`:
   - Data-related codes:`utils/minibatch.py`,`utils/preprocess.py`,`utils/random_walk.py`,`utils/utilities.py`
	- Model-related codes:`models/model.py`
  - Training validation and testing-related :`train.py`
- `data`: Our data folder. `data/DBLP3`,`data/DBLP5`.
## 3. Quick Start
### Dataset
Here, using DBLP3 and DBLP5 as examples, the following file structures are outlined: The `.npz` files contain original graph structure data. The `_deg_nc` files consist of role sets identified by a degree-based structural role discovery algorithm. The `_role_nc` files contain role sets derived from a motif-based structural role discovery algorithm, and the `_wl_nc` files hold role sets determined by a Weisfeiler-Lehman-based structural role discovery algorithm.
### Training
For link prediction
>  python train.py --task==link prediction

For node classification
>  python train.py --task==node classification
## 4. Acknowledgements
- This code is based on the works available at DySAT (https://github.com/aravindsankar28/DySAT) and EvolveGCN (https://github.com/IBM/EvolveGCN).
  
- The code will be continuously improved and updated.


