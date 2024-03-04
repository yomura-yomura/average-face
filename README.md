# Average Face

（某企業の20周年イベントで使用）


## Example
Average face of prime ministers in Japan, from 1989 to 2024.
![Japanese President's average face](/doc/average-face2.png)

How this image has been created is roughly described in the following section.
### Step 1: Align Outer Corner of Eyes by Affine Transformation
Red points in the following figure are facial landmarks predicted by dlib, trained on the [iBUG 300-W dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).
Then, the outer corners of eyes are aligned to the fixed positions by affine transformation.  
The result is shown below.
![Align Outer Corner of Eyes](/doc/similarity-transformed-images-with-eyes-aligned.png)

### Step 2: Align More Parts of Face for each Delaunay Triangle by Affine Transformation
[Delaunay Triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation), derived from the red points in the previous figure, is used to divide the face into triangles.
Then, each triangle is warped to the one of average face by affine transformation.  
The result is shown below.
![Align More Parts of Face for each Delaunay Triangle](/doc/similarity-transformed-images-with-eyes-aligned-and-warped-triangles.png)

### Step 3: Average the Aligned Images
Finally, we can get the average face like the first image by averaging the inner/outer region separately,
where the outer region is drawn in black in the figure of Step 2.

## How to Run
### Install Dependencies
Run the following command to install the required packages in the root directory of this repository.
```bash
pip install .
```
### Run the Script
TBD

## Some Issues/Limitations
- Created average face sometimes shows boarder of the face like in the first image, because of averaging the inner/outer region separately.
- Available for only frontal face images. This is limited by the current implementation of the face detection to get facial landmarks.

## Reference
- [LearnOpenCV](https://learnopencv.com/average-face-opencv-c-python-tutorial/#disqus_thread)
- [首相官邸 - 歴代内閣](https://www.kantei.go.jp/jp/rekidainaikaku/index.html)
