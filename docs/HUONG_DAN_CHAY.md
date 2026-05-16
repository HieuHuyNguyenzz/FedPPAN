# Huong Dan Chay Cac Thuat Toan

Tai lieu nay huong dan chay cac pipeline trong repo `FedPPAN`:
- `DP/fl`: Federated Learning + Gaussian Differential Privacy tren model update
- `PPAN_FL`: Federated Learning + privacy mechanism kieu PPAN
- `Local_Differential_Privacy`: Federated Learning + adaptive local DP
- `CVB_FL`: Federated Learning + Convolutional Variational Bottleneck (CVB)
- `DCS2_FL`: Federated Learning + Defense by Concealing Sensitive Samples (DCS2)

## 1) Yeu cau moi truong

- Python: khuyen nghi `3.10` hoac `3.11`
- CUDA (tuy chon): neu co GPU
- He dieu hanh: Windows/Linux deu duoc (lenh duoi day dung theo terminal Python thong thuong)

Thu vien can co (toi thieu):
- `torch`, `torchvision`
- `flwr`
- `flwr-datasets`
- `numpy`
- `scikit-learn`
- `Pillow`

## 2) Cai dat nhanh

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install torch torchvision flwr flwr-datasets numpy scikit-learn pillow
```

## 3) Cau truc chay

Tat ca pipeline deu co entrypoint rieng:
- `DP/fl/main.py`
- `PPAN_FL/main.py`
- `Local_Differential_Privacy/main.py`
- `CVB_FL/main.py`
- `DCS2_FL/main.py`

Chay tu thu muc goc repo `FedPPAN`.

## 4) Cach chay tung thuat toan

### 4.1 DP (Gaussian noise tren tham so model)

```bash
python DP/fl/main.py
```

Config tai:
- `DP/fl/config.py`

Thong so chinh:
- `NUM_CLIENTS`, `NUM_ROUNDS`, `BATCH_SIZE`, `LEARNING_RATE`
- `eplison`, `delta`, `sensitivity`

Ket qua:
- Thu muc `results/`

### 4.2 PPAN_FL

```bash
python PPAN_FL/main.py
```

Config tai:
- `PPAN_FL/config.py`

Thong so chinh:
- `PRIVACY_WEIGHT` (list)
- `NUM_CLIENTS`, `NUM_ROUNDS`, `BATCH_SIZE`, `LEARNING_RATE`, `NOISE_SCALE`

Ket qua:
- Thu muc `results/`

### 4.3 Local_Differential_Privacy

```bash
python Local_Differential_Privacy/main.py
```

Config tai:
- `Local_Differential_Privacy/config.py`

Thong so chinh:
- `INITIAL_EPSILON`, `TARGET_ACCT`, `ADJUST_RATE`, `NOISE_CLIP`, `WINDOW_SIZE`
- `NUM_CLIENTS`, `NUM_ROUNDS`, `BATCH_SIZE`, `LEARNING_RATE`

### 4.4 CVB_FL (pipeline moi)

```bash
python CVB_FL/main.py --dataset fashion --seed 42
python CVB_FL/main.py --dataset cifar --seed 42
```

Config tai:
- `CVB_FL/config.py`

Thong so CVB mac dinh theo paper:
- `CVB_POSITION=1`
- `CVB_KERNEL_SIZE=5`
- `CVB_SCALE=0.5`
- `CVB_BETA=0.1`

Ket qua:
- `results/cvb_fl/<dataset>/seed_<seed>/`

### 4.5 DCS2_FL (pipeline moi)

```bash
python DCS2_FL/main.py --dataset fashion --seed 42
python DCS2_FL/main.py --dataset cifar --seed 42
```

Config tai:
- `DCS2_FL/config.py`

Thong so DCS2 chinh:
- `DCS2_LAMBDA_G=0.7`
- `DCS2_LAMBDA_X`, `DCS2_LAMBDA_Z`, `DCS2_EPSILON`
- `DCS2_SYNTH_STEPS`, `DCS2_SYNTH_LR`
- `DCS2_INIT_MODE` (`random`)

Ket qua:
- `results/dcs2_fl/<dataset>/seed_<seed>/`

### 4.6 Protocol baseline IWQoS

CVB_FL va DCS2_FL da duoc dong bo protocol:
- Fashion-MNIST: `NUM_CLIENTS=100`, `CLIENTS_PER_ROUND=10`, `NUM_ROUNDS=200`, `LR=0.01`
- CIFAR-10: `NUM_CLIENTS=50`, `CLIENTS_PER_ROUND=10`, `NUM_ROUNDS=500`, `LR=0.03`

Metric leakage:
- `privacy_leakage` (theo metric leakage cua code goc repo)

## 5) Tong hop ket qua baseline (mean +- std)

Sau khi chay nhieu seed, tong hop bang:

```bash
python tools/summarize_baselines.py --algos cvb_fl dcs2_fl --datasets fashion cifar
```

## 6) Chay nhanh de smoke test

Khong sua config. Dung override:
- `--num-rounds 1` hoac `2`

Vi du:
```bash
python CVB_FL/main.py --dataset fashion --seed 1 --num-rounds 1
python DCS2_FL/main.py --dataset cifar --seed 1 --num-rounds 1
```

## 7) Kiem tra va debug co ban

### 6.1 Loi import/module
- Dam bao dang chay tai thu muc goc repo.
- Dam bao da cai du `flwr`, `flwr-datasets`, `torchvision`.

### 6.2 Loi tai dataset
- Lan dau se tu dong download dataset (can internet).
- Neu bi gian doan mang, chay lai lenh.

### 6.3 Chay qua cham/het RAM
- Giam `NUM_CLIENTS`
- Giam `BATCH_SIZE`
- Giam `num_gpus` trong `start_simulation(...)` neu can

## 8) Thu tu de xac nhan repo hoat dong

Nen chay theo thu tu:
1. `CVB_FL/main.py` voi 1-2 rounds (xac nhan pipeline moi)
2. `DCS2_FL/main.py` voi 1-2 rounds
3. `DP/fl/main.py` voi 1-2 rounds
4. `PPAN_FL/main.py` voi 1-2 rounds
5. `Local_Differential_Privacy/main.py` voi 1-2 rounds

Neu cac pipeline deu chay duoc voi smoke test, tang len full config de train chinh thuc.
