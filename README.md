
# CAD4LLM

## 파일 목록 및 설명

### 1. `deepcad_json_to_neutral_batch.py`

**기능**: DeepCAD 형식의 JSON 파일을 `neutral` 형식으로 변환합니다.

**사용법**:
```bash
python3 deepcad_json_to_neutral_batch.py --root <cad_json_root> --out_dir <out>
```

- `--root`: 변환할 DeepCAD JSON 파일들이 위치한 디렉터리
- `--out_dir`: 변환 후 결과를 저장할 디렉터리

**옵션**:
- `--keep_raw`: 원본 데이터를 유지하면서 변환

### 2. `whucad_vec_to_neutral.py`

**기능**: WHUCAD의 벡터 데이터를 `neutral` 형식으로 변환합니다.

**사용법**:
```bash
python3 whucad_vec_to_neutral.py --in_h5 <input_h5_file> --out_json <output_json_file>
```

- `--in_h5`: 변환할 WHUCAD 벡터 데이터가 포함된 `.h5` 파일
- `--out_json`: 변환된 데이터를 저장할 `.json` 파일

### 3. `jsonl_creator.py`

**기능**: DeepCAD 및 WHUCAD 데이터를 처리하여 JSONL 형식으로 변환합니다. 주로 훈련 데이터셋을 생성할 때 사용됩니다.

**사용법**:
```bash
python3 jsonl_creator.py
```

**설명**:
- `DATASETS`: DeepCAD 및 WHUCAD 데이터셋을 정의하고, 각 데이터셋에서 `neutral` 및 `prompt` 폴더를 찾아 데이터를 결합하여 `train.jsonl` 형식으로 변환합니다.

### 4. `ir_to_deepcad_json_rich.py`

**기능**: `cad_ir.v0.1` 형식의 IR 데이터를 DeepCAD 형식의 JSON으로 변환합니다.

**사용법**:
```bash
python3 ir_to_deepcad_json_rich.py --root <input_ir_dir> --out_dir <out_json_dir>
```

- `--root`: 변환할 IR 파일들이 위치한 디렉터리
- `--out_dir`: 변환 후 결과를 저장할 디렉터리

### 5. `neutral_to_whucad.py`

**기능**: `neutral` 형식의 CAD 데이터를 WHUCAD 형식으로 변환합니다.

**사용법**:
```bash
python3 neutral_to_whucad.py --ir_root <input_ir_dir> --out_root <output_h5_dir>
```

- `--ir_root`: 변환할 `neutral` JSON 파일들이 위치한 디렉터리
- `--out_root`: 변환 후 결과를 저장할 H5 파일들이 위치한 디렉터리

### 6. `neutral_to_xml.py`

**기능**: `neutral` 형식의 CAD 데이터를 QIF XML 형식으로 변환합니다.

**사용법**:
```bash
python3 neutral_to_xml.py --input <input_json_file_or_dir> --output <output_xml_file_or_dir>
```

- `--input`: 변환할 `neutral` JSON 파일 또는 디렉터리
- `--output`: 변환 후 저장할 QIF XML 파일 또는 디렉터리

### 7. `qwen3_run_dneutral_WHUCAD_h100.py`

**기능**: Qwen 모델을 이용하여 CAD 데이터에서 자연어 지시문을 생성하는 스크립트입니다. 이 스크립트는 주어진 `neutral` 데이터를 기반으로 CAD 작업 지시문을 자동으로 생성합니다.

**사용법**:
```bash
python3 qwen3_run_dneutral_WHUCAD_h100.py --ir_json <input_ir_file> --out_txt <output_txt_file>
```

- `--ir_json`: 변환할 IR 파일 (예: `.ir.json` 형식)
- `--out_txt`: 생성된 텍스트 지시문을 저장할 파일

**옵션**:
- `--lang`: 언어 선택 (기본값: `en`)

### 8. `qwen3_run_dneutral_h100_2.py`

**기능**: `neutral` 형식의 CAD 데이터를 기반으로 Qwen3 모델을 통해 자연어 지시문을 생성합니다. 이 파일은 대규모 배치 처리를 지원하며, 다양한 파라미터를 설정하여 생성된 지시문을 저장합니다.

**사용법**:
```bash
python3 qwen3_run_dneutral_h100_2.py --ir_root <input_ir_root> --out_root <output_txt_root>
```

- `--ir_root`: `neutral` JSON 파일들이 위치한 디렉터리
- `--out_root`: 생성된 지시문을 저장할 텍스트 파일이 위치한 디렉터리

**옵션**:
- `--batch_size`: 배치 크기 설정 (기본값: 12)
- `--max_new_tokens`: 생성할 최대 토큰 수 (기본값: 180)
- `--temperature`: 생성 온도 설정 (기본값: 0.35)
- `--top_p`: 샘플링 비율 설정 (기본값: 0.9)
- `--top_k`: 샘플링 값 설정 (기본값: 60)

### 9. `qwen3_run_dneutral_h100_2_.py`

**기능**: Qwen3 모델을 통해 CAD 데이터를 자연어로 변환하는 스크립트로, 다양한 모델과 하이퍼파라미터를 조정할 수 있습니다. 이 스크립트는 모델을 로드하고, 배치 처리를 통해 CAD 지시문을 생성합니다.

**사용법**:
```bash
python3 qwen3_run_dneutral_h100_2_.py --ir_root <input_ir_root> --out_root <output_txt_root>
```

- `--ir_root`: 변환할 `neutral` JSON 파일들이 위치한 디렉터리
- `--out_root`: 생성된 지시문을 저장할 텍스트 파일이 위치한 디렉터리

**옵션**:
- `--batch_size`: 배치 크기 설정 (기본값: 12)
- `--temperature`: 생성 온도 설정 (기본값: 0.35)
- `--top_p`: 샘플링 비율 설정 (기본값: 0.9)

---

## 사용 예시

### 1. **DeepCAD JSON을 Neutral로 변환**

```bash
python3 deepcad_json_to_neutral_batch.py --root ./DeepCAD/json_files --out_dir ./output/neutral
```

### 2. **WHUCAD 벡터 데이터 변환**

```bash
python3 whucad_vec_to_neutral.py --in_h5 ./WHUCAD/data/vec --out_json ./output/neutral_file.json
```

### 3. **Neutral 데이터를 QIF XML로 변환**

```bash
python3 neutral_to_xml.py --input ./neutral_data --output ./output/qif_data.xml
```

### 4. **Qwen3 모델을 사용하여 CAD 지시문 생성**

```bash
python3 qwen3_run_dneutral_WHUCAD_h100.py --ir_json ./input_data/ir_file.ir.json --out_txt ./output/command.txt
```

---
