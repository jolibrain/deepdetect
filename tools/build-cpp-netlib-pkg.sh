#!/bin/bash

set -x
set -o pipefail

export DEB_BUILD_OPTIONS=nocheck


pkg_name="cpp-netlib"
pkg_version="0.11.2+dfsg1"
deb_version="2"

mkdir -p build-${pkg_name}
cd build-${pkg_name}

rm -rf ${pkg_name}-${pkg_version}

dget http://archive.ubuntu.com/ubuntu/pool/universe/${pkg_name:0:1}/${pkg_name}/${pkg_name}_${pkg_version}-${deb_version}.dsc
dpkg-source -x ${pkg_name}_${pkg_version}-${deb_version}.dsc
cd ${pkg_name}-${pkg_version}
sed -i -e '/dh_auto_configure --/s/$/ -DCPP-NETLIB_BUILD_TESTS:BOOL=OFF/g' debian/rules
debuild -us
