from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import io

router = APIRouter()

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for analysis"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        return {
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "sample_data": df.head().to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
