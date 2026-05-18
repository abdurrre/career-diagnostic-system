# support dataset
Dataset ini jadi sumber **data tambahan** untuk membantu **mengurangi class imbalance** yang ada di Main Dataset. Isinya terdiri dari dataset Job Descriptions 2025 dan LinkedIn Software Engineering Jobs.

Dataset ini dipakai di notebook **Data_augmented_part_2.ipynb** dengan pendekatan cap sampling supaya distribusi tiap role jadi lebih seimbang.
Tujuannya supaya role yang datanya sedikit, seperti Frontend dan Fullstack Developer, tetap punya jumlah sampel yang cukup sehingga model NLP bisa belajar lebih adil dan tidak terlalu bias ke role mayoritas.

Note: Support Dataset sebenarnya **terdiri dari 2 dataset**, tetapi yang diunggah ke GitHub hanya 1 dataset karena **ukuran dataset lainnya terlalu besar**. Dataset tersebut disimpan terpisah melalui Google Drive.

https://drive.google.com/drive/folders/1G65uT53DOE_SvyaDLYrMGtliDcUnUMro?usp=share_link
