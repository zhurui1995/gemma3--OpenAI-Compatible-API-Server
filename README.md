# gemma3--OpenAI-Compatible-API-Server
Gemma 3 OpenAI-Compatible API Server. Gemma 3 fits openwebui.
[中文版本](readme_zh.md)
# Gemma 3 OpenAI-Compatible API Server

This project provides a FastAPI backend server that wraps the Google Gemma 3 (`gemma-3-27b-it`) large language model, exposing it through an API compatible with OpenAI's `v1/chat/completions` endpoint.

This allows you to run the Gemma 3 model locally on your own hardware and interact with it using clients designed for the OpenAI API, such as [Open WebUI](https://github.com/open-webui/open-webui).

## Features

*   **OpenAI API Compatibility:** Exposes `/v1/chat/completions` and `/v1/models` endpoints.
*   **Streaming Support:** Provides real-time, token-by-token streaming responses (`stream=True`) using Server-Sent Events (SSE).
*   **Non-Streaming Support:** Supports standard request/response cycles (`stream=False`).
*   **Gemma 3 Integration:** Uses the `transformers` library to load and run the `google/gemma-3-27b-it` model.
*   **Multi-GPU Support:** Leverages `accelerate` and `device_map="auto"` for distributing the model across multiple GPUs.
*   **Basic Multimodal Input:** Accepts image URLs or base64-encoded images in the OpenAI message format (requires `Pillow`).
*   **Configurable:** Easily change the model path, visible GPUs, host, and port.

## Requirements

### Hardware

*   **GPU:** One or more powerful NVIDIA GPUs with sufficient VRAM to run the `gemma-3-27b-it` model (likely > 40GB VRAM total, depending on quantization/configuration). CUDA capability is required.
*   **RAM:** Sufficient system RAM.
*   **Storage:** Enough disk space for the model files and Python environment.

### Software

*   **Python:** 3.8+ (Developed with 3.11)
*   **CUDA Toolkit:** Compatible version for your GPU drivers and PyTorch.
*   **Python Packages:** See `requirements.txt` or install manually (see Installation).
*   **Git:** For cloning the repository.

### Model Files

*   You need to have the `gemma-3-27b-it` model files downloaded locally. The default path configured in the script is `/home/user/t1/google/gemma-3-27b-it`. **You must update this path in the script if your model is located elsewhere.**

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install required Python packages:**
    ```bash
    pip install fastapi uvicorn pydantic requests pillow accelerate torch transformers bitsandbytes python-dotenv # Add any other specific dependencies used
    # Or, if you create a requirements.txt:
    # pip install -r requirements.txt
    ```
    *Note: Ensure your `torch` installation is compatible with your CUDA version.*

## Configuration

1.  **Model Path:**
    *   Open the main Python script (e.g., `main.py`).
    *   Locate the `MODEL_DIR` variable.
    *   Change its value to the **absolute path** where your `gemma-3-27b-it` model files are stored.
    ```python
    # --- Configuration ---
    MODEL_DIR = "/path/to/your/google/gemma-3-27b-it" # <-- UPDATE THIS
    # ... other configurations
    ```

2.  **GPU Devices:**
    *   The script uses the `CUDA_VISIBLE_DEVICES` environment variable to determine which GPUs to use. Set this variable in your terminal **before** running the server.
    *   Example (using GPUs 0, 1, 2, 4, 5, 6, 7):
        ```bash
        export CUDA_VISIBLE_DEVICES="0,1,2,4,5,6,7"
        ```
    *   If you only want to use specific GPUs, list their indices. If you have only one GPU (e.g., index 0), use `export CUDA_VISIBLE_DEVICES="0"`.

3.  **Host and Port (Optional):**
    *   You can change the `HOST` and `PORT` variables in the script if needed. The default `HOST="0.0.0.0"` allows connections from other machines on your network.

## Running the Server

1.  **Set the `CUDA_VISIBLE_DEVICES` environment variable:**
    ```bash
    export CUDA_VISIBLE_DEVICES="0,1,2,4,5,6,7" # Adjust indices as needed
    ```

2.  **Start the FastAPI server using Uvicorn:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    *   Replace `main` with the name of your Python script if it's different.
    *   The server will start, load the model (this may take some time), and listen for requests on `http://0.0.0.0:8000`.

## Usage with Clients (Example: Open WebUI)

1.  Ensure the API server is running.
2.  Open your Open WebUI instance.
3.  Navigate to **Settings** -> **Connections**.
4.  Under the "Ollama API" section (or similar for custom connections):
    *   Set the **API Base URL** to `http://<your_server_ip>:8000/v1`
        *   Replace `<your_server_ip>` with the actual IP address of the machine running the FastAPI server. If Open WebUI is on the *same* machine, you can use `localhost` or `127.0.0.1`.
        *   **Important:** Make sure to include the `/v1` suffix in the URL.
    *   Save the connection.
5.  Go back to the main chat interface.
6.  Click on the model selection dropdown.
7.  You should see the model ID (e.g., `/home/user/t1/google/gemma-3-27b-it` or the path you configured) listed. Select it.
8.  You can now chat with your locally hosted Gemma 3 model, with streaming enabled! You can also try uploading images if your client supports sending them via the OpenAI API format.

## API Endpoints

*   **`GET /v1/models`**
    *   Lists the available model(s). In this setup, it will return the configured `MODEL_DIR`.
    *   Response format mimics OpenAI's model list.

*   **`POST /v1/chat/completions`**
    *   The main endpoint for generating chat responses.
    *   Accepts JSON payloads compatible with the OpenAI Chat Completions API (including `messages`, `model`, `stream`, `max_tokens`, `temperature`, etc.).
    *   Supports `stream=True` for Server-Sent Events (SSE) streaming and `stream=False` for a single JSON response.

## Notes and Limitations

*   **Memory:** The `gemma-3-27b-it` model is large and requires significant GPU VRAM. Ensure your hardware meets the requirements.
*   **Image Handling:** Image processing is basic. It downloads or decodes images specified by URL or base64 data but doesn't perform complex analysis or adhere to OpenAI's `detail` parameter.
*   **Stop Sequences:** Custom stop sequences provided in the API request are not currently implemented in the generation logic.
*   **`n > 1`:** Generating multiple choices (`n > 1`) in a single request is not supported by the current implementation using `model.generate`.
*   **Error Handling:** Error handling is basic; further improvements could be made.
*   **`finish_reason` (Streaming):** The `finish_reason` in streaming mode is simplified.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

(Optional) Specify your license here. E.g.:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
