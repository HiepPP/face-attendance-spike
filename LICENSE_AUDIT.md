# License Audit for Commercial SaaS Use

Date: 2026-10-23. Scope: everything that the web demo (`demo/web.py`) loads at runtime, plus the Stack 2 weights on disk. This is an engineering audit, not legal advice.

## Verdict

- Code and runtime libraries: commercial use is allowed. No license purchase is needed. Keep the attribution notices.
- Model weights: the weight files have permissive licenses. But no source documents the SFace training data, and the YuNet training data is non-commercial. Get a legal review before production.
- No component needs a paid license, except the GPL video codecs in the macOS OpenCV wheel. Remove them from any shipped build (see below).

## Model weights

| Model | Weight license | Training data | Risk |
|---|---|---|---|
| YuNet (detect) | MIT | WIDER FACE (CC BY-NC-ND 4.0) | Low-medium |
| SFace (embed) | Apache-2.0 | Not documented | Medium-high |
| MiniFASNet (liveness) | Apache-2.0 | Not documented | Medium |
| dlib ResNet (Stack 2, not in app) | CC0 | VGG Face, FaceScrub, web scrape | Medium |

- SFace: the paper trains on CASIA-WebFace, VGGFace2, and MS-Celeb-1M. These sets have research-only terms or were withdrawn (Microsoft withdrew MS-Celeb-1M in 2019). Nobody has confirmed which set trained the 2021dec weights.
- SFace: issues opencv/opencv_zoo#313 (open since 2026-07-22) and #318 (open since 2026-10-01) ask this exact question for the same sha256. The only maintainer reply (#318) says: "The Apache 2 license should be clear enough for your usage". zhongyy/SFace#9 has no reply.
- YuNet: libfacedetection.train (BSD-3-Clause) trains on WIDER FACE. The Hugging Face dataset card lists WIDER FACE as CC BY-NC-ND 4.0. Whether a model trained on non-commercial data inherits the restriction is legally unsettled. YuNet ships inside OpenCV and is widely used commercially.
- MiniFASNet: the repo is Apache-2.0. The README does not name the training data. Minivision also sells a more accurate closed model (TPR 99.7% vs 97.8% open source).

## Code and libraries (runtime)

- OpenCV 4.14 (Apache-2.0), ONNX Runtime (MIT), NumPy (BSD-3-Clause and permissive). The Python stdlib web server needs nothing more.
- Bench-only: pandas, scikit-learn, matplotlib (BSD or PSF-based). Dev-only: torch, onnx (BSD/Apache). The full list from installed metadata has only permissive licenses.
- Finding: the macOS `opencv-python` 4.14.0.94 wheel bundles FFmpeg linked to `libx264` and `libx265` (GPL-2.0+). The wheel's `LICENSE-3RD-PARTY.txt` does not list them. The Linux `manylinux_2_28` wheels (`opencv-python` and `opencv-python-headless`) do not contain x264 or x265.
- The pipeline uses no video I/O. For a server, use `opencv-python-headless` on Linux. For Android, use the OpenCV Android SDK. If you ship a macOS or desktop build, build OpenCV with `-DWITH_FFMPEG=OFF`.
- A SaaS that only runs on your own servers does not distribute the code. So GPL and LGPL obligations do not apply there. An Android or desktop app does distribute the code, so its licenses screen must show every notice.

## Not used in the app

- `face_recognition` (MIT) is not used. Its model package bundles the forbidden 68-point dlib model, and the clone that held it was deleted.
- No InsightFace, EdgeFace, YOLO-Face, AdaFace, or ArcFace weights are present.

## Options to reduce model risk

1. Accept the residual risk after legal review. Record the review in `THIRD_PARTY_NOTICES.md`.
2. Ask the open issues for written confirmation of the SFace training data (#313 and #318 already ask this).
3. Train a face embedding model on data with clear commercial rights. This costs data and GPU time.
4. Buy a commercial face recognition SDK that comes with license terms and an indemnity.

## Other compliance (not licenses)

- Face data is sensitive personal data under Vietnamese data protection law. Check consent, retention, and processing terms with legal before production.

## Sources

- https://github.com/opencv/opencv_zoo/issues/313
- https://github.com/opencv/opencv_zoo/issues/318
- https://github.com/zhongyy/SFace (paper abstract lists the training sets)
- https://github.com/ShiqiYu/libfacedetection.train (trains YuNet on WIDER Face)
- https://huggingface.co/datasets/CUHK-CSE/wider_face (license: cc-by-nc-nd-4.0)
- https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (README: open-source vs closed model)
- Local: `.venv/.../cv2/.dylibs/libavcodec*.dylib` links `libx264` and `libx265` (`otool -L`)
