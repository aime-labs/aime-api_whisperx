import argparse
import logging
import io
import base64
import torch
import json
import whisperx
import contextlib
from pydub.utils import mediainfo
import torchaudio
import wave
import gc
import os
import tempfile
from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "whisper_x"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_name", type=str, default="large-v2", help="WhisperX model size")
    parser.add_argument("--auth_key", type=str, default=DEFAULT_WORKER_AUTH_KEY, help="API worker authentication key")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for transcription")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size for VAD segments in seconds")
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
        job_id = job_data.get('job_id')

        if 'audio_input' in job_data:
            base64_data = job_data['audio_input'].split(',')[1]
            
            audio_data = base64.b64decode(base64_data)

            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(audio_data)
                temp_file_path = f.name
            
            audio = whisperx.load_audio(temp_file_path)

            # Trancription
            result = model.transcribe(
                audio, 
                batch_size=args.batch_size, 
                print_progress=args.print_progress,
            )
            logger.info(f"Transcribtion complete")
            logger.info(result)

            # Alignment
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            align_result = whisperx.align(
                result["segments"], model_a, metadata, audio, device, return_char_alignments=False
            )

            logger.info(f"Alignment complete")

            # Memory management: deleting models to free GPU resources
            del model_a
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Diarization
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token="hf_QyjnDmmxaMYlqTXaJPLXqGUkPtQWFjlYrBe", device=device
            )
            diarize_segments = diarize_model(audio)
            logger.info(f"Diarization complete")
            diarization_result = whisperx.assign_word_speakers(diarize_segments, align_result)
            diarization_result = json.dumps(diarization_result)
            logger.info(diarization_result)       

            transcription_text = ' '.join(segment['text'] for segment in result['segments'])

            output = {
                'result': transcription_text,
                'diarization_result': diarization_result,
            }
            
            api_worker.send_job_results(output, job_id=job_id)
            logger.info(f"Job {job_id} processed successfully.")

        else:
            logger.warning("No audio input received.")
            api_worker.send_job_results({'error': 'No audio input'}, job_id=job_id)

if __name__ == "__main__":
    main()
