mkdir data
cd data
wget https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip
unzip dataset_small_v1.1.zip

if [ ! -f "ShapeNet/metadata.yaml" ]; then
    cp metadata.yaml ShapeNet/metadata.yaml
fi
