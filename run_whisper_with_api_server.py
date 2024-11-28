import argparse
import logging
import base64
import torch
import json
import tempfile
import gc
import os
from aime_api_worker_interface import APIWorkerInterface
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "whisperx")))
import whisperx

WORKER_JOB_TYPE = "whisper_x"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_name", type=str, default="large-v3", help="WhisperX model size")
    parser.add_argument("--auth_key", type=str, default=DEFAULT_WORKER_AUTH_KEY, help="API worker authentication key")
    parser.add_argument("--print_progress", action="store_true", help="Print progress during transcription")
    return parser

def main():
    parser = argparse.ArgumentParser(description="WhisperX inference with transcription, alignment, and speaker diarization")
    parser.add_argument("--api_server", type=str, default="http://0.0.0.0:7777", help="API server address")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use")
    parser = add_inference_arguments(parser)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    logger.info(f"Running inference on {device} with compute type {compute_type}.")

    model = whisperx.load_model(args.model_name, device, compute_type=compute_type)

    api_worker = APIWorkerInterface(
        args.api_server, WORKER_JOB_TYPE, args.auth_key, args.gpu_id,
        world_size=1, rank=0, gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    )

    while True:
        job_data = api_worker.job_request()
        print("Data received")
        job_id = job_data.get('job_id')

        if 'audio_input' in job_data:
            base64_data = job_data['audio_input'].split(',')[1]
            audio_data = base64.b64decode(base64_data)
            print("Data ready")

            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(audio_data)
                temp_file_path = f.name
            
            print("Temp file ready")
            audio = whisperx.load_audio(temp_file_path)
            
            
            lang = None if job_data.get('src_lang') == "none" else job_data.get('src_lang')
            chunk_size = int(job_data.get('chunk_size'))

            # Trancription
            result = model.transcribe(
                audio,
                print_progress = True,
                language = lang,
                chunk_size = chunk_size,
            )
            logger.info(f"Transcription complete")
            logger.info(result)

            # Alignment
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            align_result = whisperx.align(
                result["segments"], model_a, metadata, audio, device, return_char_alignments=False, print_progress = True,
            )
            align_result = json.dumps(align_result)
            logger.info(f"Alignment complete")

            # Memory management
            del model_a
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()     

            transcription_text = ' '.join(segment['text'] for segment in result['segments'])

            output = {
                'result': transcription_text,
                'align_result': align_result,
            }
            
            api_worker.send_job_results(output, job_id=job_id)
            logger.info(f"Job {job_id} processed successfully.")

        else:
            logger.warning("No audio input received.")
            api_worker.send_job_results({'error': 'No audio input'}, job_id=job_id)

if __name__ == "__main__":
    main()
