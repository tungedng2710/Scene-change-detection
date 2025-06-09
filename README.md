# Scene Change Detection
This project provides a small FastAPI service that detects changes between two images.
It uses the [Segment Any Change](https://github.com/Z-Zheng/pytorch-change-models) model to produce
an overlay mask and can optionally describe the differences with a vision language model.

<p align="center">
  <img src="[static/demo.png](https://github.com/tungedng2710/Scene-change-detection/blob/main/app/static/demo.png?raw=true)" alt="Descriptive text for your image" width="500"/>
</p>

## Features

- REST API built with FastAPI
- Change detection using pre-trained Segment Any Change weights
- Optional description of changes via Gemini or Ollama
- Dockerfile and docker-compose configuration for GPU inference

## Installation

Create a Python virtual environment and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Some packages (e.g. PyTorch models) can take a while to install. Alternatively,
you can use the provided `Dockerfile` or `docker-compose.yaml` for a containerised setup.

## Running the API

To launch the API locally after installing the requirements:

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 7866 --reload
```

Open your browser at `http://localhost:7866` to access the demo page.

### Docker

If you prefer Docker, build and run the image:

```bash
docker-compose up --build
```

## API Usage

Send a `POST` request to `/detect` with two image files (`ref_img` and `test_img`).
The response contains logs, the percentage of changed pixels and a link to the
mask overlay image.

Example using `curl`:

```bash
curl -F "ref_img=@ref.png" -F "test_img=@test.png" http://localhost:7866/detect
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
