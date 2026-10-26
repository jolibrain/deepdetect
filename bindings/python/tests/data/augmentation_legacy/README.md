Fixtures captured before the replay refactor, with OpenCV 4.6.0. `reference.cc`
uses the original `ImgRandAugCV` implementation, BGR input, seed-matched native
and OpenCV RNGs, and `augment_with_segmap`. Images are PNGs; all three fixture
outputs and their masks are checked pixel-for-pixel. Regenerating with a changed
implementation is not a substitute for testing backward compatibility.
