NAME=$1
PREFIX=$2
COMPLETION_STAMP=$3
SOURCE_POSTFIX=$4
BUILD_POSTFIX=$5
J=$6
CMAKE_COMMAND=$7
shift 7
ARGS=$@
[ -f "$COMPLETION_STAMP" ] && echo "$NAME" already built, not rebuilding && exit 0
echo "External build of : $NAME"
echo "with prefix       : $PREFIX"
echo "completion stamp  : $COMPLETION_STAMP"
echo "source postfix    : $SOURCE_POSTFIX"
echo "build postfix     : $BUILD_POSTFIX"
echo "-j                : $J"
echo "CMAKE_COMMAND     : $CMAKE_COMMAND"
echo "additional args   : $ARGS"
$CMAKE_COMMAND -S $PREFIX/$SOURCE_POSTFIX -B $PREFIX/$BUILD_POSTFIX $ARGS
make -C $PREFIX/$BUILD_POSTFIX -j$J
