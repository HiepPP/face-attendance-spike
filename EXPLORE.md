# Spike Exploration Notes

Date: 2026-10-23. Sources are shallow clones in `upstream/` (gitignored, code only, no LFS weights).

## Upstream sources

| Repo | Path in `upstream/` | License |
|---|---|---|
| opencv/opencv_zoo (sparse: YuNet, SFace) | `opencv_zoo/` | YuNet MIT, SFace Apache-2.0 |
| minivision-ai/Silent-Face-Anti-Spoofing | `Silent-Face-Anti-Spoofing/` | Apache-2.0 |
| ageitgey/face_recognition | `face_recognition/` | MIT |
| ageitgey/face_recognition_models | `face_recognition_models/` | CC0 (bundles forbidden model) |
| davisking/dlib (sparse) | `dlib/` | BSL-1.0 |
| davisking/dlib-models | `dlib-models/` | CC0 |

## Blocker: Stack 2 via `face_recognition` pulls a forbidden model

- `face_recognition` depends on `face_recognition_models>=0.3.0` (`face_recognition/setup.py`).
- That package ships `shape_predictor_68_face_landmarks.dat` (99.7 MB).
- `face_recognition/api.py:19-20` loads the 68-point model at import time, even with `model="small"`.
- So `pip install face_recognition` breaks handoff section 2.2.
- Proposed workaround: call `dlib` directly. Use `dlib.get_frontal_face_detector()` (HOG), `shape_predictor_5_face_landmarks.dat`, and `dlib_face_recognition_resnet_model_v1.dat`. Download only those two files from dlib-models.
- Needs reviewer approval: the handoff lists `face_recognition` as the Stack 2 wrapper.

## YuNet (detect)

- Weight `face_detection_yunet_2023mar.onnx`: sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`, 232589 bytes (from LFS pointer).
- The 2023mar model has a fixed input shape. OpenCV 4.x reshapes it per call. Plain ONNX Runtime does not.
- A new `face_detection_yunet_2026may.onnx` (sha256 `ebafce4e...f0f0`) has dynamic H/W. It is the better choice for ONNX Runtime on Android.
- Output row: `[x, y, w, h, 10 landmark coords, score]` (15 values). `alignCrop` takes this row directly.
- The model detects faces of about 10-300 px. At tablet resolution, downscale frames first.

## SFace (embed)

- Weight `face_recognition_sface_2021dec.onnx`: sha256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`, 38696353 bytes.
- Reference thresholds: cosine 0.363, norm-L2 1.128 (`sface.py:24-25`).
- `alignCrop` warps to the 112x112 5-point template inside OpenCV.

## MiniFASNet (liveness)

- Weights are in the repo, not LFS: `resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth` and `4_0_0_80x80_MiniFASNetV1SE.pth`.
- Output has 3 classes. Class index 1 is "real" (`test.py:63-64`). Score is the summed softmax divided by 2.
- Input is BGR, HWC to CHW, float in range 0-255. Upstream removed the `/255` step (`src/data_io/functional.py:59-60`). The ONNX export and Android code must not normalize.
- Crop: scale bbox around its center, clamp scale to the image, shift the box inside the frame (`src/generate_patches.py`). Port this exactly.
- Scale is parsed from the file name: `2.7` and `4_0_0` mean 4.0 (`src/utility.py:parse_model_name`).
- Upstream requires 3:4 (W:H) input frames, to match Android camera frames (`test.py:25-31`).
- Upstream uses a RetinaFace Caffe detector for the bbox. The spike uses YuNet boxes. This is a deviation: box size can differ and shift the liveness score. Record it in the report.
- Upstream pins torch 1.2. A newer torch is likely to load the state dicts. Strip the `module.` prefix as `anti_spoof_predict.py` does.

## Android implications

- With ONNX Runtime only, YuNet post-processing and SFace alignment need a Kotlin port. The other option is the OpenCV Android SDK (Apache-2.0), which gives `FaceDetectorYN` and `FaceRecognizerSF` directly.
- Recommendation: use OpenCV Android SDK for detect and align and embed. Use ONNX Runtime for MiniFASNet.

## Licensing notes

- dlib ResNet weights are CC0. The README says the training data includes VGG Face and FaceScrub. Mention this in `THIRD_PARTY_NOTICES.md` for reviewer awareness.
- Silent-Face repo is Apache-2.0. It also ships a RetinaFace Caffe model. The spike does not need it.
