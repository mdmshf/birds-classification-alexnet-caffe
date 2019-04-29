
EXAMPLE=/scratch/digits/deps/caffe/examples/birds
DATA=/scratch/digits/deps/caffe/examples/birds/input
TOOLS=/scratch/digits/deps/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/input/train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
