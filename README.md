# PaddleOCR Web Stack

Full-stack image-to-text service powered by PaddleOCR.

## Getting Started

In production/deployment:
* The frontend primarily depends on React 19 and Rolldown Vite 7.2:
    1. Install dependencies.
        ```shell
        npm install
        ```
    2. Configure the backend URI.
       ```shell
       cp .env.sample .env
       # and update .env
       ```
    3. Deploy the frontend.
        ```shell
        npm run host
        ```
* The backend depends on Docker compose v5 and optionally NVIDIA Docker runtime.
    ```shell
    docker compose up -d --build
    ```
