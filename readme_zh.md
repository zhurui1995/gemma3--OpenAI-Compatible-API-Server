# Gemma 3 OpenAI 兼容 API 服务器

本项目提供了一个 FastAPI 后端服务器，它封装了 Google 的 Gemma 3 (`gemma-3-27b-it`) 大语言模型，并通过与 OpenAI `v1/chat/completions` 端点兼容的 API 将其暴露出来。

这使你能够在自己的硬件上本地运行 Gemma 3 模型，并使用为 OpenAI API 设计的客户端（例如 [Open WebUI](https://github.com/open-webui/open-webui)）与其交互。

## 主要特性

*   **OpenAI API 兼容:** 提供 `/v1/chat/completions` 和 `/v1/models` 端点。
*   **流式响应支持:** 使用服务器发送事件 (SSE) 提供实时的、逐字（token）的流式响应 (`stream=True`)。
*   **非流式响应支持:** 支持标准的请求/响应周期 (`stream=False`)。
*   **Gemma 3 集成:** 使用 `transformers` 库加载并运行 `google/gemma-3-27b-it` 模型。
*   **多 GPU 支持:** 利用 `accelerate` 和 `device_map="auto"` 将模型分布到多个 GPU 上。
*   **基本多模态输入:** 接受 OpenAI 消息格式中的图像 URL 或 base64 编码的图像（需要 `Pillow` 库）。
*   **可配置:** 轻松更改模型路径、可见的 GPU、主机和端口。

## 环境要求

### 硬件

*   **GPU:** 一块或多块性能强劲的 NVIDIA GPU，具有足够的显存 (VRAM) 来运行 `gemma-3-27b-it` 模型（根据量化/配置，总显存可能需要 > 40GB）。需要支持 CUDA。
*   **内存 (RAM):** 足够的系统内存。
*   **存储:** 足够的磁盘空间用于存放模型文件和 Python 环境。

### 软件

*   **Python:** 3.8+ (本项目使用 3.11 开发)
*   **CUDA Toolkit:** 与你的 GPU 驱动和 PyTorch 版本兼容的 CUDA 工具包。
*   **Python 依赖包:** 查看 `requirements.txt` 文件，或手动安装（见安装部分）。
*   **Git:** 用于克隆代码仓库。

### 模型文件

*   你需要**预先下载** `gemma-3-27b-it` 模型文件到本地。脚本中配置的默认路径是 `/home/user/t1/google/gemma-3-27b-it`。**如果你的模型存放在其他位置，必须在脚本中更新此路径。**

## 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone <你的仓库URL>
    cd <你的仓库名称>
    ```

2.  **创建并激活虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows 系统请使用 `venv\Scripts\activate`
    ```

3.  **安装所需的 Python 包:**
    ```bash
    pip install fastapi uvicorn pydantic requests pillow accelerate torch transformers bitsandbytes python-dotenv # 添加你使用的其他特定依赖
    # 或者，如果你创建了 requirements.txt 文件:
    # pip install -r requirements.txt
    ```
    *注意: 请确保你的 `torch` 安装版本与你的 CUDA 版本兼容。*

## 配置说明

1.  **模型路径:**
    *   打开主 Python 脚本 (例如 `main.py`)。
    *   找到 `MODEL_DIR` 变量。
    *   将其值更改为你存放 `gemma-3-27b-it` 模型文件的**绝对路径**。
    ```python
    # --- Configuration ---
    MODEL_DIR = "/path/to/your/google/gemma-3-27b-it" # <-- 更新此路径
    # ... 其他配置
    ```

2.  **GPU 设备:**
    *   脚本使用 `CUDA_VISIBLE_DEVICES` 环境变量来决定使用哪些 GPU。请在**运行服务器之前**在你的终端设置此环境变量。
    *   示例 (使用 GPU 0, 1, 2, 4, 5, 6, 7):
        ```bash
        export CUDA_VISIBLE_DEVICES="0,1,2,4,5,6,7"
        ```
    *   如果你只想使用特定的 GPU，请列出它们的索引。如果你只有一个 GPU (例如索引为 0)，使用 `export CUDA_VISIBLE_DEVICES="0"`。

3.  **主机和端口 (可选):**
    *   如果需要，你可以更改脚本中的 `HOST` 和 `PORT` 变量。默认的 `HOST="0.0.0.0"` 允许来自你网络中其他机器的连接。

## 运行服务器

1.  **设置 `CUDA_VISIBLE_DEVICES` 环境变量:**
    ```bash
    export CUDA_VISIBLE_DEVICES="0,1,2,4,5,6,7" # 根据需要调整索引
    ```

2.  **使用 Uvicorn 启动 FastAPI 服务器:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    *   如果你的 Python 脚本文件名不是 `main.py`，请替换 `main`。
    *   服务器将启动，加载模型（这可能需要一些时间），并在 `http://0.0.0.0:8000` 上监听请求。

## 如何与客户端配合使用 (以 Open WebUI 为例)

1.  确保 API 服务器正在运行。
2.  打开你的 Open WebUI 实例。
3.  导航到 **设置 (Settings)** -> **连接 (Connections)**。
4.  在 "Ollama API" 部分（或其他自定义连接部分）：
    *   将 **API 基础 URL (API Base URL)** 设置为 `http://<你的服务器IP>:8000/v1`
        *   将 `<你的服务器IP>` 替换为运行 FastAPI 服务器的机器的实际 IP 地址。如果 Open WebUI 运行在**同一台**机器上，你可以使用 `localhost` 或 `127.0.0.1`。
        *   **重要:** 确保 URL 末尾包含 `/v1` 后缀。
    *   保存连接。
5.  返回主聊天界面。
6.  点击模型选择下拉菜单。
7.  你应该能看到模型 ID（例如 `/home/user/t1/google/gemma-3-27b-it` 或你配置的路径）被列出。选择它。
8.  现在你可以与本地托管的 Gemma 3 模型聊天了，并且流式输出是启用的！如果你的客户端支持通过 OpenAI API 格式发送图片，你也可以尝试上传图片。

## API 端点

*   **`GET /v1/models`**
    *   列出可用的模型。在此设置中，它将返回配置的 `MODEL_DIR`。
    *   响应格式模仿 OpenAI 的模型列表。

*   **`POST /v1/chat/completions`**
    *   用于生成聊天响应的主要端点。
    *   接受与 OpenAI Chat Completions API 兼容的 JSON 负载（包括 `messages`, `model`, `stream`, `max_tokens`, `temperature` 等）。
    *   支持 `stream=True`（用于 SSE 流式传输）和 `stream=False`（用于单个 JSON 响应）。

## 注意事项与限制

*   **显存:** `gemma-3-27b-it` 模型非常大，需要大量 GPU 显存。请确保你的硬件满足要求。
*   **图像处理:** 图像处理功能比较基础。它会下载或解码由 URL 或 base64 数据指定的图像，但不会执行复杂的分析或遵循 OpenAI 的 `detail` 参数。
*   **停止序列:** API 请求中提供的自定义停止序列 (stop sequences) 当前未在生成逻辑中实现。
*   **`n > 1`:** 当前使用 `model.generate` 的实现不支持在单个请求中生成多个选项 (`n > 1`)。
*   **错误处理:** 错误处理比较基础，可以进一步改进。
*   **`finish_reason` (流式):** 流式模式下的 `finish_reason` 被简化了。

## 贡献

欢迎贡献！随时提出 issue 或提交 Pull Request。

## 许可证

(可选) 在此处指定你的许可证。例如：
本项目采用 MIT 许可证授权 - 详情请参阅 [LICENSE](LICENSE) 文件。
