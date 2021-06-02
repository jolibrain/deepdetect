NAME=$1
PREFIX=$2
DOWNLOAD_STAMP=$3
PATCH_STAMP=$4
SOURCE_LOCATION=$5
shift 5
PATCHES=$@
[ -f "$PREFIX/$DOWNLOAD_STAMP" ] && [ -f "$PREFIX//$PATCH_STAMP" ] && [ $(( $(stat -c %Y "$PREFIX/$PATCH_STAMP") - $(stat -c %Y "$PREFIX/$DOWNLOAD_STAMP")  )) -ge 0 ] && echo "$NAME" already patched, skipping && exit 0
echo "External patch of : $NAME"
echo "with prefix       : $PREFIX"
echo "download stamp    : $DOWNLOAD_STAMP"
echo "patch stamp       : $PATCH_STAMP"
echo "source location   : $SOURCE_LOCATION"
echo "patches           : $PATCHES"
cd $SOURCE_LOCATION
git apply $PATCHES
