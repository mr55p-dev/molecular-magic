# Number of instances

| Type    | Source         | N instances |
| ------- | -------------- | ----------- |
| Unclean | original MolE8 | 55,418      |
| Clean   | original MolE8 | 42,471      |
| Unclean | magic          | 55,272      |
| Clean   | magic          | 42,468      |

# QM9 Filtering
Filtering QM9 leads to removing 118,247 instances being removed and 14,642 instances being written.
This means 10.9% of the database is usable according to MolE8 rules. The removal process is broken down
as follows:
| Reason               | Num. mols removed |
| -------------------- | ----------------- |
| Other atom           | 2.163             |
| Heavy atom           | 109,769           |
| Zero free energy     | 0                 |
| Long bond            | 169               |
| Strained angle       | 5,675             |
| Tetravalent nitrogen | 158               |
| Carbanion            | 313               |
| Lone hydrogen        | 3                 |
