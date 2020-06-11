# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script should be run from the root of tensorflow_graphics folder.

#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

export PYTHONPATH="$PWD:$PYTHONPATH"

pushd tensorflow_graphics/projects/local_implicit_grid/

pip install tensorflow
pip install -r requirements.txt


wget https://storage.googleapis.com/local-implicit-grids/pretrained_ckpt.zip
unzip  pretrained_ckpt.zip && rm pretrained_ckpt.zip

mkdir -p demo_data
wget https://cs.uwaterloo.ca/~c2batty/bunny_watertight.obj
mv -f bunny_watertight.obj demo_data

python resample_geometry.py \
--input_mesh=demo_data/bunny_watertight.obj \
--output_ply=demo_data/bunny_pts.ply

python reconstruct_geometry.py \
--input_ply=demo_data/bunny_pts.ply \
--npoints=2048 --steps=3001 --part_size 0.1 --output_ply=/tmp/output.ply

popd
