
# CAD4LLM
https://drive.google.com/drive/folders/137db_VVZ09NnmAlIaSRz4e0Fzj7f4L0S?usp=sharing
<img width="1330" height="62" alt="image" src="https://github.com/user-attachments/assets/79cdfd2f-6745-4815-85bb-12b255b17c4e" />


## 파일 목록 및 설명

`/Qwen` : LLM 관련 작업 디렉토리

`/converter` : 중립포맷으로 변환하는데 사용하는 converter (하단 설명 참조)

`/dataset` : 최종 출력된 중립포맷(Neutral)과 UserPrompt 디렉토리 (최종 데이터셋 폴더)

### `deepcad_json_to_neutral_batch.py`

**기능**: DeepCAD 형식의 JSON 파일을 `neutral` 형식으로 변환

**사용법**:
```bash
python3 deepcad_json_to_neutral_batch.py --root <cad_json_root> --out_dir <out>
```

- `--root`: 변환할 DeepCAD JSON 파일들이 위치한 디렉토리
- `--out_dir`: 변환 후 결과를 저장할 디렉토리

**옵션**:
- `--keep_raw`: 원본 데이터를 유지하면서 변환

### `h5_to_neutral.py`

**기능**: 벡터 데이터를 `neutral` 형식으로 변환

**사용법**:
```bash
python3 h5_to_neutral.py --in_h5 <input_h5_file> --out_json <output_json_file>
```

- `--in_h5`: 변환할 WHUCAD 데이터가 포함된 `.h5` 파일
- `--out_json`: 변환된 데이터를 저장할 `.json` 파일

### `jsonl_creator.py`

**기능**: DeepCAD 및 WHUCAD 데이터를 처리하여 JSONL 형식으로 변환

**사용법**:
```bash
python3 jsonl_creator.py
```

**설명**:
- `DATASETS`: 각 데이터셋에서 `neutral` 및 `prompt` 폴더를 찾아 데이터를 결합하여 `train.jsonl` 형식으로 변환

### `neutral_to_whucad.py`

**기능**: `neutral` 형식의 CAD 데이터를 WHUCAD 형식으로 변환

**사용법**:
```bash
python3 neutral_to_whucad.py --neutral_root <input_neutral_dir> --out_root <output_h5_dir>
```

- `--neutral_root`: 변환할 `neutral` JSON 파일들이 위치한 디렉토리
- `--out_root`: 변환 후 결과를 저장할 h5 파일들이 위치한 디렉토리

### `neutral_to_xml.py`

**기능**: `neutral` 형식의 CAD 데이터를 XML 형식으로 변환

**사용법**:
```bash
python3 neutral_to_xml.py --input <input_json_file_or_dir> --output <output_xml_file_or_dir>
```

- `--input`: 변환할 `neutral` JSON 파일 또는 디렉토리
- `--output`: 변환 후 저장할 XML 파일 또는 디렉토리

### `qwen3_run_dneutral_WHUCAD_h100.py`

**기능**: Qwen 모델을 이용하여 CAD 데이터에서 UserPrompt를 생성 (WHUCAD). `neutral` 데이터를 기반으로 UserPrompt 생성

**사용법**:
```bash
python3 qwen3_run_dneutral_WHUCAD_h100.py --neutral_json <input_neutral_file> --out_txt <output_txt_file>
```

- `--neutral_json`: 변환할 neutral 파일 `.neutral.json` 형식
- `--out_txt`: 생성된 텍스트 지시문을 저장할 파일

**옵션**:
- `--lang`: 언어 선택 (en, ko)

### `qwen3_run_dneutral_h100.py`

**기능**: `neutral` 형식의 CAD 데이터를 Qwen3 모델을 통해 자연어 지시문 생성. (h100 , torchrun 전용)

**사용법**:
```bash
python3 qwen3_run_dneutral_h100.py --neutral_root <input_neutral_root> --out_root <output_txt_root>
```
```bash
torchrun --standalone --nproc_per_node=4 qwen3_run_dneutral_h100.py --neutral_root <input_neutral_root>  --out_root <output_txt_root>
```

- `--neutral_root`: `neutral` JSON 파일들이 위치한 디렉토리
- `--out_root`: 생성된 지시문을 저장할 텍스트 파일이 위치한 디렉토리

**옵션**:
- `--batch_size`: 배치 크기 설정 (기본값: 12)
- `--max_new_tokens`: 생성할 최대 토큰 수 (기본값: 180)
- `--temperature`: 생성 temperature 설정 (기본값: 0.35)
- `--top_p`: 샘플링 비율 (기본값: 0.9)
- `--top_k`: 샘플링 값 (기본값: 60)
---

## 사용 예시

### **DeepCAD JSON을 Neutral로 변환**

```bash
python3 deepcad_json_to_neutral_batch.py --root ./DeepCAD/json_files --out_dir ./output/neutral
```

### **WHUCAD 벡터 데이터를 Neutral로 변환**

```bash
python3 whucad_vec_to_neutral.py --in_h5 ./WHUCAD/data/vec --out_json ./output/neutral_file.json
```

### **Neutral 데이터를 XML로 변환**

```bash
python3 neutral_to_xml.py --input ./neutral_data --output ./output/qif_data.xml
```

---
