#!/bin/bash -x
# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

JOB_NAME="pointnet_`date +"%b%d_%H%M%S"`"
PROJECT_ID=${1:-`gcloud config get-value project`}
LOGDIR=${2:-$TENSORBOARD_DEFAULT_LOGDIR}
REGION="us-central1" #< WARNING: match region of bucket!

# see: https://cloud.google.com/ai-platform/training/docs/using-gpus
MACHINE_TYPE="--scale-tier custom --master-machine-type standard_v100"


# --- checks a logdir has been set, exits otherwise
[ -z "$LOGDIR" ] && echo "Logdir not specified." && exit 1

# --- The container configuration
cat > /tmp/Dockerfile <<EOF
  FROM tensorflow/tensorflow:2.1.0-gpu-py3

  # --- Install git
  RUN apt-get update
  RUN apt-get install -y git

  # --- Install TFG dependencies
  RUN apt-get -y install libopenexr-dev
  RUN apt-get -y install libgles2-mesa-dev
  RUN apt-get -y install libc-ares-dev

  # --- Install dependencies
  RUN pip3 install --upgrade pip
  RUN pip3 install numpy
  RUN pip3 install matplotlib
  RUN pip3 install tqdm
  RUN pip3 install wget
  RUN pip3 install h5py
  RUN pip3 install absl-py
  RUN pip3 install tensorflow_datasets

  # --- Copy source tree (recursive)
  COPY . /

  # --- Install TFG
  # WORKDIR /
  # RUN sh build_pip_pkg.sh
  # RUN pip3 install --upgrade dist/*.whl

  # --- Execute
  WORKDIR /tensorflow_graphics/projects/pointnet
  ENTRYPOINT ["python3", "train.py"]
EOF


if [ "${1}" == "local" ]
then
  # --- Launches the job locally
  TAG="local_pointnet"
  docker build -f /tmp/Dockerfile -t $TAG $PWD/../../../
  docker run --gpus all $TAG --logdir $LOGDIR

else
  # --- Launches the job on aiplatform
  TAG="gcr.io/$PROJECT_ID/pointnet"
  docker build -f /tmp/Dockerfile -t $TAG $PWD/../../../
  docker push $TAG
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --master-image-uri $TAG \
    $MACHINE_TYPE \
    -- \
    --job_name $JOB_NAME \
    --logdir $LOGDIR

  # --- Streams the job logs to local terminal
  gcloud ai-platform jobs stream-logs $JOB_NAME
fi
