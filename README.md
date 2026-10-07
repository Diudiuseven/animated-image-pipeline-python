# animated-image-pipeline-python

本地、进程内的服务端图片加工链路：方向纠正 → 色彩归一 → 缩放 → 静态/动图转码。

## 快速开始

```bash
uv sync
uv run pytest          # 72 项自动化测试
uv run python demo.py  # 生成样本并转码到 ./demo_out/，校验确定性
uv run python main.py  # 启动 HTTP 服务 http://127.0.0.1:8000
```

## 示例

```bash
curl -F "file=@anim.gif;type=image/gif" \
     -F "output_format=webp" -F "width=320" \
     http://127.0.0.1:8000/images/process -o out.webp
```

详细的格式、帧语义、配置项、错误码与并发/取消说明见
[`docs/README.md`](docs/README.md)。

## 模块

- `image_pipeline.validation` 早期拒绝（签名/参数/上界）
- `image_pipeline.codec_io` 确定性解码/编码总装
- `image_pipeline.gif_reader` 无状态 GIF 容器解析（原始 tile/透明/处置）
- `image_pipeline.apng_reader` 无状态 APNG 解析（未预乘原始帧）
- `image_pipeline.webp_reader` WebP ANMF 帧元数据解析
- `image_pipeline.color` ICC→sRGB 归一（直 alpha，不产生黑边）
- `image_pipeline.orientation` EXIF 方向纠正（含动图 tile 变换）
- `image_pipeline.render` 整数 alpha 合成（disposal/blend，无残影）
- `image_pipeline.transform` 单一滤波器确定性缩放
- `image_pipeline.pipeline` 编排（校验→解码→合成→缩放→编码）
- `image_pipeline.concurrency` 有界并发/排队/取消/超时
- `image_pipeline.server` FastAPI 上传/下载接口
- `image_pipeline.samples` 进程内确定性样本生成
