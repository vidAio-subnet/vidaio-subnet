# video_retrieval

## 📁 Project Structure

```plaintext
project-root/
├── modules/                # Search Engine
│   |── config.py
|   |── db.py
|   |── metadata_uploader.py
|   |── search_engine.py
|   └── searchy.py
├── config.yaml             # configuration for db setting
│   
├── requirements.txt        # dependency
│   
├── gen_uploader.py         # db generation for metadata
├── gen_db.py               # hash feature generation
├── test_search.py          # search test
│   
└── README.md               # Project overview and documentation
```

## Instruction
This is video retrieval system using perception hash feature.
```plaintext
- Main feature
. Frame-wise perception hash
. coarse search
. fine search
```

## Running
### Predepency installation
```plaintext
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### How to run
```plaintext
fill out config.yaml
python gen_uploader.py --dir videodb
python gen_db.py --dir videodb --feature original
python test_search.py --query_path query.mp4
```