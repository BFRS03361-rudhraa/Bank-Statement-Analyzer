import os
import argparse
import redis
from rq import Queue, Worker
from multiprocessing import Process
from worker_redis import parse_pdf  # import the worker-safe function

def _run_worker_process(default_timeout=2400):
    """
    Start an RQ worker in this process and begin processing jobs.
    """
    redis_conn = redis.Redis()
    q = Queue(connection=redis_conn, default_timeout=default_timeout)
    worker = Worker([q])
    worker.work()

# --- Enqueue function ---
def enqueue_jobs(input_path, output_dir="./output", max_pages=0,job_timeout=2400):
    redis_conn = redis.Redis()
    q = Queue(connection=redis_conn)

    jobs = []

    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        job = q.enqueue(parse_pdf, input_path, output_dir, max_pages, job_timeout=job_timeout)
        print(f"Enqueued {input_path} -> job ID {job.id}")
        jobs.append(job)

    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(input_path, file)
                job = q.enqueue(parse_pdf, pdf_path, output_dir, max_pages, job_timeout=job_timeout)
                print(f"Enqueued {pdf_path} -> job ID {job.id}")
                jobs.append(job)
    else:
        raise ValueError(f"Input {input_path} is not a valid PDF or folder.")

    return jobs

# --- Function to start multiple workers ---
def start_workers(num_workers=4, default_timeout=2400):
    redis_conn = redis.Redis()
    q = Queue(connection=redis_conn, default_timeout=default_timeout)
    worker_processes = []

    for _ in range(num_workers):
        p = Process(target=_run_worker_process, args=(default_timeout,))
        p.start()
        worker_processes.append(p)

    return worker_processes

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDFs in parallel using Redis + RQ")
    parser.add_argument("input", help="Path to PDF file or folder")
    parser.add_argument("--out", default="./output", help="Output directory")
    parser.add_argument("--pages", type=int, default=0, help="Max pages to process (0 = all)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Step 1: Enqueue PDFs
    enqueue_jobs(args.input, args.out, args.pages)

    # Step 2: Start workers
    worker_processes = start_workers(args.workers)
    print(f"Started {args.workers} workers. Processing jobs...")

    # Step 3: Wait for workers to finish
    try:
        for p in worker_processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping workers...")
        for p in worker_processes:
            p.terminate()
