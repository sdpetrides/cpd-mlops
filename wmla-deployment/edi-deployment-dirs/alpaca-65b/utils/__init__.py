import subprocess

from datetime import datetime, timezone


def install_requirements(dir_python_pkg):

    t0 = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %z')
    print(f"{t0}: Installing Libraries in {dir_python_pkg} ...")

    for package_name in [
        "boto3==1.28.2",
        "botocore==1.31.2",
        "requests==2.29.0",
        "urllib3==1.26.16",
    ]:
        out = subprocess.check_output(
            f"pip install {package_name} --no-deps --target {dir_python_pkg}",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
        print(out)

    for package_name in [
        "accelerate==0.21.0",
        "bitsandbytes==0.39.1",
        "peft==0.4.0",
        "tokenizers==0.13.3",
        "torch==2.0.1",
        "transformers==4.31.0",
    ]:

        out = subprocess.check_output(
            f"pip install {package_name} --target {dir_python_pkg}",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
        print(out)