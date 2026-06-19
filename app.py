import os
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from networksecurity.logging.logger import logger
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.pipeline.batch_prediction import BatchPrediction
from networksecurity.constants.training_pipeline import MODEL_FILE_NAME

FINAL_MODEL_PATH = os.path.join("final_model", MODEL_FILE_NAME)

app = FastAPI(
    title="Network Security — Phishing Detection API",
    description="Train the ML pipeline or run batch predictions on network data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templetes")


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def home(request: Request):
    return templates.TemplateResponse(request, "table.html", {"results": None})


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "Network Security API is running"}


@app.post("/train", tags=["Training"])
def train_pipeline():
    """Trigger the full training pipeline (ingestion → validation → transformation → model training)."""
    try:
        logger.info("API: training pipeline triggered")
        artifact = TrainingPipeline().run_pipeline()
        return {
            "status": "success",
            "trained_model_path": artifact.trained_model_file_path,
            "train_metrics": {
                "f1_score":        artifact.train_metric_artifact.f1_score,
                "precision_score": artifact.train_metric_artifact.precision_score,
                "recall_score":    artifact.train_metric_artifact.recall_score,
            },
            "test_metrics": {
                "f1_score":        artifact.test_metric_artifact.f1_score,
                "precision_score": artifact.test_metric_artifact.precision_score,
                "recall_score":    artifact.test_metric_artifact.recall_score,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_class=HTMLResponse, tags=["Prediction"])
async def batch_predict(request: Request, file: UploadFile = File(...)):
    """Upload a CSV file of network features and render prediction results as an HTML table."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    if not os.path.exists(FINAL_MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Run /train first.",
        )

    temp_path = f"temp_upload_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        predictor = BatchPrediction(input_file_path=temp_path, model_path=FINAL_MODEL_PATH)
        output_path = predictor.initiate_batch_prediction()

        result_df = pd.read_csv(output_path)
        predictions = result_df.to_dict(orient="records")

        return templates.TemplateResponse(request, "table.html", {"results": predictions})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
