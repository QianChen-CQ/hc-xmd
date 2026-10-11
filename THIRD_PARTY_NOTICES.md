# Third-party sources and checkpoints

The root license applies to newly authored HC-XMD code and documentation. It
does not relicense upstream source, pretrained weights, or separately supplied
experimental artifacts. Source manifests record the exact retrieved snapshots.

| Component | Source and pinned commit | Treatment |
| --- | --- | --- |
| RegretNet 2019 reference | [saisrivatsan/deep-opt-auctions](https://github.com/saisrivatsan/deep-opt-auctions), `466e154b123921c142ee811216d0e64415bf31a5` | The HC-XMD PyTorch port cites the MIT-licensed reference; its notice is retained at `phase1/REGRETNET_UPSTREAM_LICENSE.txt`. |
| AMenuNet reference | [Haoran0301/AMenuNet](https://github.com/Haoran0301/AMenuNet), `a62ec44e9280ec7a87313d1b8036ccfa5402091e` | Independently written paper implementation. Upstream source is not redistributed; no explicit license was found in the inspected snapshot. |
| Optimaler source and released checkpoints | [dimonenka/optimaler](https://github.com/dimonenka/optimaler), `1f35c1e27f62bfa18bcd610de2a553156f9e6da5` | Distributed with the upstream MIT notice at `vendor/optimaler/LICENSE`. Exact source/checkpoint hashes are in `OPTIMALER_MANIFEST.json`. No separate checkpoint license was present in the archived snapshot; the HC-XMD root license makes no additional grant for those weights. |
| CITransNet network | [zjduan/CITransNet](https://github.com/zjduan/CITransNet), `b33e34058d1e3febe1f0fbeb1d5d799c268cd928` | MIT notice retained at `phase3_author/vendor/CITransNet/LICENSE`; hashes in `phase3_author/SOURCES.json`. |
| PreferenceNet network definitions and README | [neeharperi/PreferenceNet](https://github.com/neeharperi/PreferenceNet), `1419a4654a4c6a5aabffd7eb2a89d077979cd965` | **Not redistributed here.** No explicit repository license was found in the inspected snapshot. `tools/fetch_preferencenet.py` retrieves the three exact files directly from upstream, or copies the user's local snapshot, and verifies SHA-256. They are excluded from Git. |
| GemNet and ProportionNet algorithms | Papers linked in `gemnet/IMPLEMENTATION_NOTES.txt` and `phase3_author/SOURCES.json` | Documented paper-directed implementations. ProportionNet and its matched baseline use the external PreferenceNet RegretNet architecture, so the external setup step is required. |

Retrieving publicly accessible upstream files does not itself grant permission
to republish them. Do not remove the three PreferenceNet exclusions from
`.gitignore` or add those downloaded files to a public release without resolving
their upstream permissions. Fetching them does not change the experiment code.

The bridge checks every entry in `phase3_author/SOURCES.json` before building a
Phase III model, so this external setup is required for all Phase III methods,
including the context-only methods. The exact archived files are required;
using a newer default-branch version will fail the hash checks by design.
