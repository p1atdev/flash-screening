# Flash Screening

https://arxiv.org/html/2604.01178v3 の再現実装の高速化

./arXiv-2604.01178v3 に tex source あり。なければ https://arxiv.org/src/2604.01178v3 を wget して解凍して。

## 高速化

src/flash_screening/eager.py に既に pytorch 実装がある。

これを参考にして、より高速に動作するカーネル版を作成する。

