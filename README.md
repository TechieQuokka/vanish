# Vanish - 오디오 노이즈 제거 시스템

<div align="center">

**깨끗한 음성을 보존하는 AI 기반 오디오 노이즈 제거 시스템**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/yourusername/vanish)

[기능](#기능) • [설치](#설치) • [빠른 시작](#빠른-시작) • [문서](#문서) • [성능](#성능)

</div>

## 🎵 샘플 오디오 다운로드

처리된 오디오 샘플을 다운로드하여 Vanish의 노이즈 제거 성능을 직접 확인해보세요:

- 📥 [원본 오디오 (MP3)](result/Shinya%20Aoki%20vs.%20Yoshihiro%20Akiyama%20%20ONE%20Championship%20Full%20Fight%20-%20ONE%20Championship.mp3) - 24.98 MB
- 📥 [노이즈 제거 완료 (WAV)](result/Shinya%20Aoki%20vs.%20Yoshihiro%20Akiyama%20%20ONE%20Championship%20Full%20Fight%20-%20ONE%20Championship_clean.wav) - 55.09 MB

> **참고**: 처리된 WAV 파일은 무손실 품질로 제공되므로 파일 크기가 더 큽니다.

---

## 개요

Vanish는 오디오 녹음에서 배경 소음(TV 소리, 주변 소음, 교통 소음 등)을 제거하면서 깨끗하고 자연스러운 음성을 보존합니다. 최첨단 딥러닝 모델을 기반으로 하며 NVIDIA GPU에 최적화되어 있습니다.

### 아키텍처

```
입력 오디오 → 전처리 → 음원 분리 (Demucs) →
음성 향상 (Resemble-Enhance) → 후처리 → 깨끗한 음성
```

### 핵심 기술

- **Demucs v4**: 음원 분리를 위한 하이브리드 트랜스포머
- **Resemble-Enhance**: 확산 모델 기반 음성 향상
- **VoiceFixer**: 대체 음성 향상 모델
- **PyTorch**: GPU 가속 처리
- **RTX 3060 12GB 최적화**

## 기능

✅ **다단계 파이프라인**
- Demucs v4를 사용한 음원 분리
- Resemble-Enhance를 통한 음성 향상
- 전문가급 후처리

✅ **고품질 출력**
- SNR 개선: 일반적으로 20+ dB
- PESQ 점수: 3.5+ (지각적 품질)
- STOI 점수: 0.9+ (명료도)

✅ **GPU 가속**
- CPU 처리 대비 6배 빠름
- RTX 3060 12GB 최적화
- 자동 메모리 관리

✅ **유연한 구성**
- 품질 프리셋 (빠름/균형/고품질)
- YAML 기반 설정
- 배치 처리 지원

✅ **다양한 인터페이스**
- 명령줄 인터페이스 (CLI)
- Python API
- 구성 가능한 파이프라인

✅ **포맷 지원**
- 입력: WAV, MP3, FLAC, M4A/AAC
- 출력: WAV (16/24-bit PCM)
- 자동 포맷 변환

## 설치

### 빠른 설치 (Ubuntu + Python 3.11.9 + RTX 3060)

```bash
# 1. CUDA 12.1과 함께 PyTorch 설치
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Vanish 설치
git clone https://github.com/yourusername/vanish.git
cd vanish
pip install -r requirements.txt
pip install -e .

# 3. Resemble-Enhance 설치 (선택사항, 권장)
pip install git+https://github.com/resemble-ai/resemble-enhance.git

# 4. 설치 확인
vanish info
```

**자세한 설치 지침은 [INSTALL.md](INSTALL.md)를 참조하세요**

## 빠른 시작

### 명령줄 인터페이스

```bash
# 기본 사용법
vanish input.wav -o output.wav

# 상세 출력과 함께 고품질 모드
vanish input.wav -o output.wav --quality high -vv

# 품질 메트릭 표시
vanish input.wav -o output.wav --metrics

# 배치 처리
vanish batch ./inputs ./outputs --pattern "*.wav"

# 사용자 정의 설정 사용
vanish input.wav -o output.wav --config config.yaml
```

### Python API

```python
from vanish import AudioPipeline

# 파이프라인 초기화
pipeline = AudioPipeline()

# 오디오 파일 처리
result = pipeline.process('input.wav', 'output.wav')

# 결과 확인
print(f"출력: {result.output_path}")
print(f"처리 시간: {result.processing_time:.2f}초")
print(f"SNR: {result.metrics.snr:.2f} dB")
print(f"PESQ: {result.metrics.pesq:.2f}")
print(f"STOI: {result.metrics.stoi:.3f}")
```

### 사용자 정의 설정

```python
from vanish import AudioPipeline, PipelineConfig

# YAML에서 설정 로드
config = PipelineConfig.from_yaml('config.yaml')

# 또는 사용자 정의 설정 생성
config = PipelineConfig(
    device='cuda',
    enhancement_mode='resemble',
    save_intermediate=True
)

# 고품질 설정
config.demucs.shifts = 2
config.resemble.denoiser_run_steps = 50

# 초기화 및 처리
pipeline = AudioPipeline(config)
result = pipeline.process('input.wav', 'output.wav')
```

## 성능

### RTX 3060 12GB (1분 오디오)

| 품질 모드 | 처리 시간 | GPU 메모리 | 품질 (SNR) |
|----------|----------|-----------|-----------|
| **빠름** | 15-20초 | ~4-5 GB | 18-22 dB |
| **균형** | 20-30초 | ~6-7 GB | 22-26 dB |
| **고품질** | 40-60초 | ~7-8 GB | 26-30 dB |

### CPU vs GPU 비교 (1분 오디오)

| 플랫폼 | 시간 | 속도 향상 |
|--------|------|----------|
| CPU (8코어) | 2-3분 | 1배 |
| RTX 3060 12GB | 20-30초 | **6배** |
| RTX 4090 24GB | 10-15초 | **12배** |

**최적화 팁은 [docs/PERFORMANCE.md](docs/PERFORMANCE.md)를 참조하세요**

## 문서

### 빠른 링크
- 📚 [빠른 시작 가이드](docs/QUICKSTART.md) - 5분 안에 시작하기
- 🏗️ [아키텍처](docs/ARCHITECTURE.md) - 시스템 설계 및 구성요소
- ⚡ [성능 가이드](docs/PERFORMANCE.md) - RTX 3060 최적화 팁
- 💻 [설치](INSTALL.md) - 상세 설정 지침
- 📊 [프로젝트 요약](docs/PROJECT_SUMMARY.md) - 전체 개요

### 예제
- [Python API 예제](examples/basic_usage.py)
- [CLI 예제](examples/cli_examples.sh)
- [설정 예제](config.yaml)

## 설정

### 설정 파일 생성

```bash
# 기본 설정 생성
vanish create-config config.yaml --preset rtx3060
```

### 설정 옵션

```yaml
# 장치 및 품질 설정
device: cuda
quality_mode: balanced  # fast, balanced, high

# 음원 분리 (Demucs)
demucs:
  model: htdemucs
  shifts: 1  # 0=빠름, 1=균형, 2=고품질
  segment: 10  # 초

# 음성 향상 (Resemble-Enhance)
resemble:
  denoiser_run_steps: 30  # 15=빠름, 30=균형, 50=고품질
  enhance_run_steps: 30

# 후처리
postprocess:
  noise_gate_threshold: -40  # dB
  target_lufs: -16  # 라우드니스
  highpass_cutoff: 80  # Hz
  apply_deessing: true
```

## 시스템 요구사항

### 최소
- Python 3.9+
- 8GB RAM
- 6GB GPU VRAM (또는 CPU)
- 2GB 저장공간

### 권장 (현재 구현)
- Python 3.11.9
- Ubuntu (WSL2 호환)
- NVIDIA RTX 3060 12GB
- 16GB RAM
- 5GB 저장공간
- CUDA 12.1

### 프로덕션
- Python 3.11+
- NVIDIA RTX 4090 / A100
- 32GB RAM
- 10GB NVMe SSD
- CUDA 12.1+

## 개발

### 개발 환경 설정

```bash
# 개발 의존성과 함께 설치
make dev-install

# 테스트 실행
make test

# 코드 포맷팅
make format

# 린팅 실행
make lint

# 모든 검사 실행
make check
```

### 프로젝트 구조

```
vanish/
├── src/vanish/          # 메인 패키지
│   ├── modules/         # 처리 모듈
│   ├── config.py        # 설정
│   ├── pipeline.py      # 파이프라인 오케스트레이션
│   └── cli.py           # 명령줄 인터페이스
├── tests/               # 테스트 스위트
├── examples/            # 사용 예제
├── docs/                # 문서
└── config.yaml          # 기본 설정
```

## 문제 해결

### CUDA 메모리 부족

```yaml
# config.yaml에서 메모리 사용량 줄이기
demucs:
  segment: 5  # 10에서 감소

resemble:
  chunk_size: 220500  # 441000에서 감소
```

### 느린 처리

```bash
# GPU 사용 확인
vanish info

# 빠른 모드 사용
vanish input.wav -o output.wav --quality fast
```

### 설치 문제

자세한 문제 해결은 [INSTALL.md](INSTALL.md)를 참조하세요

## 기여

기여를 환영합니다! 다음 단계를 따라주세요:

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 감사의 글

### 모델 및 라이브러리
- [Demucs](https://github.com/facebookresearch/demucs) by Facebook Research
- [Resemble-Enhance](https://github.com/resemble-ai/resemble-enhance) by Resemble AI
- [VoiceFixer](https://github.com/haoheliu/voicefixer) by Hao Heliu
- PyTorch, Librosa 및 기타 오픈소스 라이브러리

### 연구 논문
- Hybrid Transformers for Music Source Separation (Demucs v4)
- Speech Enhancement with Diffusion Models (Resemble-Enhance)
- VoiceFixer: Speech Restoration with Generative Models

## 인용

연구나 프로젝트에서 Vanish를 사용하는 경우:

```bibtex
@software{vanish2024,
  title={Vanish: Audio Noise Removal System},
  author={Vanish Team},
  year={2024},
  version={1.0.0},
  url={https://github.com/yourusername/vanish}
}
```

## 지원

- 📖 [문서](docs/)
- 💡 [예제](examples/)
- 🐛 [이슈](https://github.com/yourusername/vanish/issues)
- 💬 [토론](https://github.com/yourusername/vanish/discussions)

## 로드맵

### v1.1 (단기)
- [ ] 실시간 처리
- [ ] 웹 인터페이스 (Gradio)
- [ ] 추가 품질 프리셋
- [ ] 개선된 오류 처리

### v2.0 (중기)
- [ ] 화자 분리
- [ ] 언어 감지
- [ ] 클라우드 배포
- [ ] 모바일 SDK

### v3.0 (장기)
- [ ] 사용자 정의 모델 학습
- [ ] 비디오 지원
- [ ] AI 기반 복원

---

<div align="center">

**PyTorch와 최첨단 AI 모델로 만들어졌습니다 ❤️**

[⬆ 맨 위로](#vanish---오디오-노이즈-제거-시스템)

</div>
