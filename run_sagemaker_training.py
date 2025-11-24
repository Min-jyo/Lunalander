import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
import time
from datetime import datetime

def run_training_job(script_name, job_name_prefix, sagemaker_session, role):
    """SageMaker PyTorch Estimator를 생성하고 학습 작업을 시작합니다."""
    
    # 현재 시간을 기반으로 고유한 작업 이름 생성
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"{job_name_prefix}-{timestamp}"
    
    print(f"Starting training job: {job_name}")
    print(f"Entry script: {script_name}")

    estimator = PyTorch(
        entry_point=script_name,
        source_dir='./src',  # 'src' 폴더를 소스 디렉터리로 지정
        role=role,
        instance_count=1,
        instance_type='ml.c5.xlarge',
        framework_version='2.0', # PyTorch 버전
        py_version='py310',      # Python 버전
        # 하이퍼파라미터는 스크립트 내에 하드코딩되어 있으므로 여기서는 비워둡니다.
        # hyperparameters={},
        sagemaker_session=sagemaker_session,
        # SageMaker가 생성하는 기본 출력 경로 사용
    )

    # 학습 작업 시작 (비동기적으로 실행됨)
    estimator.fit(job_name=job_name, wait=False)
    
    return estimator, job_name

def wait_for_job_and_download(estimator, job_name, local_path):
    """학습 작업이 완료될 때까지 기다리고 결과물을 다운로드합니다."""
    
    print(f"Waiting for job '{job_name}' to complete...")
    # 학습 작업이 완료될 때까지 대기
    estimator.latest_training_job.wait()
    
    # 학습 작업의 S3 출력 경로 가져오기
    s3_output_path = estimator.model_data
    
    # S3 경로 파싱
    s3_uri_parts = s3_output_path.replace("s3://", "").split("/")
    bucket_name = s3_uri_parts[0]
    # 'output/model.tar.gz' 부분을 'output/results.npz'로 변경
    key = "/".join(s3_uri_parts[1:-1]) + "/results.npz"

    print(f"Job '{job_name}' completed. Downloading results from s3://{bucket_name}/{key}")

    try:
        # S3에서 결과 파일 다운로드
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, key, local_path)
        print(f"Successfully downloaded results to '{local_path}'")
    except Exception as e:
        print(f"Error downloading results for job {job_name}: {e}")
        print("Please check the S3 path and permissions.")


def main():
    # --- 사전 준비 ---
    # AWS 인증 정보가 로컬에 설정되어 있어야 합니다. (e.g., via `aws configure`)
    # 실행 권한이 있는 IAM 역할(role) ARN이 필요합니다.
    
    sagemaker_session = sagemaker.Session()
    # 기본 S3 버킷 사용, 없으면 SageMaker가 생성
    bucket = sagemaker_session.default_bucket() 
    
    # 본인의 IAM 역할을 여기에 입력하세요.
    # 예: "arn:aws:iam::123456789012:role/SageMaker-ExecutionRole"
    # 아래 코드는 현재 세션에서 역할을 자동으로 가져오려고 시도합니다.
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        print("Error: Could not automatically get IAM role.")
        print("Please replace `sagemaker.get_execution_role()` with your SageMaker execution role ARN string.")
        return

    print(f"Using bucket: {bucket}")
    print(f"Using role: {role}")

    # --- 1. DQN 및 Dueling DQN 학습 작업 시작 ---
    dqn_estimator, dqn_job_name = run_training_job(
        'dqn_lander.py', 'dqn-lunar-lander', sagemaker_session, role
    )
    
    dueling_estimator, dueling_job_name = run_training_job(
        'dueling_dqn_lander.py', 'dueling-dqn-lunar-lander', sagemaker_session, role
    )

    # --- 2. 각 작업이 완료될 때까지 기다리고 결과 다운로드 ---
    wait_for_job_and_download(dqn_estimator, dqn_job_name, 'dqn_results.npz')
    wait_for_job_and_download(dueling_estimator, dueling_job_name, 'dueling_dqn_results.npz')

    # --- 3. 로컬에서 결과 분석 스크립트 실행 안내 ---
    print("\n--- All training jobs completed and results downloaded. ---")
    print("Now, you can run the plotting script locally:")
    print("\npython plot_results.py --dqn_results dqn_results.npz --dueling_results dueling_dqn_results.npz\n")


if __name__ == '__main__':
    main()
