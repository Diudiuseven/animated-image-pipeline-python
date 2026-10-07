# Animated Image Pipeline（本地图片加工服务）

一个**完全在本进程内运行**的服务端图片加工链路：接收上传图片 →
方向纠正 → 色彩归一（统一 sRGB）→ 按请求缩放 → 静态/多帧动图转码 → 下载。
不依赖任何外部服务；所有输入用本地样本或进程内生成的图即可复现。

## 1. 支持的格式

| 角色 | 格式 | 说明 |
|------|------|------|
| 输入（读取） | PNG / APNG、GIF、WebP（含动图）、JPEG、BMP、TIFF | 按魔数识别，非扩展名 |
| 输出（写出） | PNG（含 APNG）、GIF、WebP（含动图）、JPEG | JPEG 仅支持单帧 |

- 静态：PNG、JPEG、BMP、TIFF、单帧 WebP。
- 多帧：GIF、APNG、animated WebP。
- 不支持的容器 / 无法识别的签名在**解码前**被拒绝。

## 2. 帧语义（强一致保证）

转码前后以下语义保持一致，**不丢帧、不错帧、不凭空补帧**：

- 帧数（`frame_count`）
- 每帧时长（`duration_ms`，整数毫秒）
- 循环次数（`loop_count`，`0` 表示无限循环）
- 逐帧画面（按帧间处置与混合方式还原）

### 处置（disposal）
- `NONE`：帧保留在画布上。
- `BACKGROUND`：显示后把该帧覆盖区域恢复为透明背景（会清掉此前残留）。
- `PREVIOUS`：恢复到该帧绘制前的画布快照。

### 混合（blend）
- `SOURCE`：用本帧 tile 直接替换该区域（GIF、WebP do-not-blend）。
- `OVER`：以本帧 tile 的直 alpha（straight/unassociated alpha）做
  source-over 合成（APNG alpha-blend、半透明 WebP）。

### 透明与残影
- 全程使用**直 alpha + 整数运算**合成，全透明像素贡献为 0，
  因此不会产生黑边 / 颜色溢出。
- 缩放始终在 RGBA 空间进行（不压平到任何底色），半透明边缘无黑色 fringe。
- `BACKGROUND/PREVIOUS` 严格清除/恢复，上一帧不会残留为“残影”。
- APNG 使用自研**无状态解析器**直接读取未预乘的原始帧像素
  （Pillow 的 seek 会累积合成并对半透明像素做缩放，不能用于保真还原）。
- GIF 使用自研**无状态容器解析器**读取每帧的原始索引 tile、
  局部/全局调色板、透明索引、时长与处置，再交给本合成器。
- WebP 通过 RIFF `ANMF` chunk 自行解析每帧 offset/disposal/blend/时长，
  像素仍由 libwebp 解码。

## 3. 方向纠正与色彩归一（对每一帧成立）

- 方向：读取 EXIF Orientation（1..8），在缩放**之前**把方向烘焙进像素，
  并剥离方向标签。支持翻转/旋转，且对动图的 tile 偏移做一致变换。
- 色彩：所有帧统一转换到 **sRGB、8-bit RGBA**。
  - 无 ICC 的图像按 Web 默认视作 sRGB，不做隐式转换。
  - 嵌入 ICC 时以**相对色度意图 + 黑点补偿**转换到固定 sRGB。
  - 是否“已经是 sRGB”通过比较矩阵/色料与白点判定（不信任描述文字），
    因此不同色彩空间（如 Display P3 与 sRGB）的同一物理内容归一后一致。
  - alpha 通道不参与颜色转换，透明区域永不变黑。

## 4. 逐字节确定性

输出**只由源图内容 + 请求参数决定**：

- 单一固定缩放滤波器（Lanczos）。
- 固定编码参数（不写时间戳，不依赖线程顺序）。
- 同一输入在任意并发度、任意处理顺序、清缓存/重启后重新处理，
  结果 **SHA-256 完全一致**。
- `request_id` 等定位字段不参与编码字节。

## 5. 早期拒绝（可区分、可定位）

任何非法输入在**开始处理前**被拒绝，绝不产出半成品，也不会因单帧损坏
而静默丢帧。错误为稳定枚举码，附带 `request_id` 与 `source_id`：

| code | 触发条件 |
|------|----------|
| `EMPTY_INPUT` | 空字节 |
| `UNSUPPORTED_FORMAT` | 签名不识别 / 输出格式不支持 / 多帧转 JPEG |
| `TRUNCATED_OR_UNDECODABLE` | 截断、无法解码、CRC/LZW/帧数据损坏 |
| `INVALID_FRAME_METADATA` | 时长非法/缺失、尺寸非法、tile 越界 |
| `FRAME_COUNT_EXCEEDED` | 帧数超上界 |
| `PIXEL_LIMIT_EXCEEDED` | 单帧/画布/输出像素超上界 |
| `INVALID_REQUEST` | 字节数超界、宽高/quality/timeout 参数非法 |
| `DIMENSION_TOO_LARGE` | 目标宽高超过 `max_dimension` |
| `QUEUE_FULL` | 已达在制容量，调用方应稍后重试（HTTP 503） |
| `TIMEOUT` | 超过请求截止时间（HTTP 504） |
| `CANCELLED` | 客户端取消（HTTP 499） |
| `INTERNAL` | 兜底，仍带定位字段 |

## 6. 容量上界与并发

全部在进程内有界，个别超大/超长动图不会拖垮整批：

- `max_decode_workers` / `max_encode_workers`：解码/编码工作并发上界。
- `max_inflight`：被接纳的在制请求总数；超出立即 `QUEUE_FULL`（有界内存）。
- `max_frame_count`、`max_frame_pixels`、`max_canvas_pixels`：
  单图帧数与像素上界。
- CPU 密集阶段在有界线程池中执行，事件循环不被阻塞。
- 请求之间**无共享可变状态**，失败/取消不会污染后续请求。

## 7. 取消、超时与半成品隔离

- 每个请求都有截止时间（`timeout_ms`，缺省 `default_timeout_ms`）。
- 可被 asyncio 任务取消；取消/超时后只返回结构化错误，
  不暴露任何半成品字节，也不留下影响后续请求的中间状态。
- 失败原因可定位到具体 `request_id` + `source_id`。

## 8. 配置项（`image_pipeline.config`）

`Limits`（均为进程内显式硬上界）：

| 字段 | 默认 | 含义 |
|------|------|------|
| `max_bytes` | 64 MiB | 允许的编码后输入大小 |
| `max_frame_count` | 500 | 单图最大帧数 |
| `max_frame_pixels` | 64,000,000 | 单帧 tile 最大像素 |
| `max_canvas_pixels` | 64,000,000 | 合成画布/输出最大像素 |
| `min/max_frame_duration_ms` | 0 / 60,000 | 合法帧时长范围 |
| `max_dimension` / `min_dimension` | 16,384 / 1 | 目标宽高边界 |
| `max_decode_workers` | 2 | 解码并发 |
| `max_encode_workers` | 2 | 编码并发 |
| `max_inflight` | 16 | 在制请求上界 |
| `queue_capacity` | 64 | 排队/重试提示容量 |
| `default_timeout_ms` | 30,000 | 默认截止时间 |

`PipelineConfig` 编码默认：`webp_quality=80`、`webp_lossless=False`、
`png_compress_level=6`、`jpeg_quality=85`、`gif_colors=256`、
目标色彩空间 `sRGB`。

## 9. HTTP API（本地）

启动：`uv run python main.py`（默认 `127.0.0.1:8000`）

- `GET /health`：状态与当前上界。
- `POST /images/process`：`multipart/form-data`（字段 `file`），
  表单参数 `output_format`、`width`、`height`、`resize_mode`、
  `quality`、`lossless`、`timeout_ms`。
- `POST /images/process-raw?output_format=webp&width=32...`：
  原始 body 即图片字节。

成功：`200` + `Content-Type: image/*`，响应头带 `X-Request-Id`、
`X-Frame-Count` 与下载文件名。失败：`4xx/5xx` JSON
`{code,message,request_id,source_id}`。

`resize_mode`：`fit`（默认，保比例内嵌）、`exact`（强制宽高可能变形）、
`cover`（居中裁剪后精确填满）。宽高给 0 表示该维度自动/保持原尺寸。

## 10. 编程用法

```python
from image_pipeline.pipeline import process_or_reject
from image_pipeline.model import ProcessRequest

result = process_or_reject(ProcessRequest(
    source=gif_bytes,
    source_id="a.gif",
    output_format="webp",   # png/gif/webp/jpeg
    width=320,              # 0 = 保持
    request_id="req-123",
))
if hasattr(result, "data"):
    open("out.webp", "wb").write(result.data)
else:
    print(result.code, result.message, result.request_id, result.source_id)
```

## 11. 测试

```
uv run pytest
```

覆盖：方向纠正、跨色彩空间归一、多帧透明/残影合成、逐字节确定性
（并发/乱序/重启）、超大动图与并发取消/超时、损坏输入、HTTP API。
所有样本由固定常量/固定随机种子在进程内生成，本地稳定复现。
