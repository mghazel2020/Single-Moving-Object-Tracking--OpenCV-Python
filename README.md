# Single Moving Object Tracking uisng OpenCV-Python

<img src="images/object-tracking-opencv.jpg" width="1000"/>

## 1. Objectives

The objective of this project is to demonstrate single moving object tracking using OpenCV-Python built-in tracking functionalities. We illustrate the development process step by step and illustrate the object tracking results.

## 2. Object Tracking

Object tracking is the task of taking an annotated or detected object of interest, localized within a bounding-box region of interest, and creating a unique ID for it, and then tracking the object location (bounding-box), as it moves around frames in a video, maintaining the ID assignment. 

* We shall demonstrate the tracking of a detected moving object using the following built-in OpenCV Python tracking functionalities:

  * Optical flow
  * Dense optical flow
  * Mean-Shift
  * Cam-Shift
  * BOOSTING 
  * MIL
  * KCF
  * TLD
  * MEDIAN FLOW

Details documentations of these tracking algorithms can be found in the references below.

## 3. Data

The input is a video with a cyclist, who represents our object of interest, as illustrated in the figure below:

<img src="images/frame-0-bbox-overlaid.jpg" width="1000"/>

## 4. Development

In this section, we shall develop the object tracking algorithms using OpenCV Python and illustrate sample tracking results:

* Author: Mohsen Ghazel (mghazel)
* Date: April 7th, 2021
* Project: Object Tracking:

The objective of this project is to demonstrate the tracking of a single localized moving object using built-in OpenCV Python tracking functionalities:

  * Optical flow
  * Dense optical flow
  * Mean-Shift
  * Cam-Shift
  * BOOSTING 
  * MIL
  * KCF
  * TLD
  * MEDIAN FLOW

We shall assume the following:
  * The moving camera is moving
  * The object of interest is also moving
  * We have the bounding box location of the object of interest from the first video frame.
  * Our object is to track the object of interest in the remaining frames.

### 4.1. Step 1: Imports and global variables


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>image <span style="color:#200080; font-weight:bold; ">as</span> mpimg

<span style="color:#595979; "># input/output OS</span>
<span style="color:#200080; font-weight:bold; ">import</span> os 

<span style="color:#595979; "># date-time to show date and time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime

<span style="color:#595979; "># Use %matplotlib notebook to get zoom-able &amp; resize-able notebook. </span>
<span style="color:#595979; "># - This is the best for quick tests where you need to work interactively.</span>
<span style="color:#44aadd; ">%</span>matplotlib notebook

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>

### 4.2. Step 2: Read the input video:

* We shall assume the following:

  * The camera is moving
  * The single object of interest is also moving
  * We have the bounding box location of the object of interest from the first video frame.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Open camera video file</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># the source video file name</span>
video_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../resources/videos/cyclist.mp4"</span>
<span style="color:#595979; "># check if the reference image file exists</span>
<span style="color:#200080; font-weight:bold; ">if</span><span style="color:#308080; ">(</span>os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>exists<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Video file name DOES NOT EXIST! = '</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># open the video file</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot read video file: "</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># get the number of frames in the video file</span>
num_video_frames <span style="color:#308080; ">=</span> <span style="color:#400000; ">int</span><span style="color:#308080; ">(</span>cap<span style="color:#308080; ">.</span>get<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>CAP_PROP_FRAME_COUNT<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Input video file: {0} has {1} frames."</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">,</span> num_video_frames<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

<span style="color:#400000; ">Input</span> video <span style="color:#400000; ">file</span><span style="color:#308080; ">:</span> <span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#44aadd; ">/</span>resources<span style="color:#44aadd; ">/</span>videos<span style="color:#44aadd; ">/</span>cyclist<span style="color:#308080; ">.</span>mp4 has <span style="color:#008c00; ">3254</span> frames<span style="color:#308080; ">.</span>
</pre>

### 4.3. Step 3: Grab the first frame of the video for manual annotation of the object of interest:

* Read the video frames
* Save the first frame to manually annotate the object of interest:
  * Object of interest annotation: the four corners of the bounding-box
    * TLC = (x1, y1)
    * TRC = (x2, y2)
    * BRC = (x3, y3)
    * BLC = (x4, y4)


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># grab and save the first video frame.</span>
<span style="color:#200080; font-weight:bold; ">for</span> counter <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">#--------------------------------------------</span>
    <span style="color:#595979; "># Step 1: read the next video frame</span>
    <span style="color:#595979; ">#--------------------------------------------</span>
    ret<span style="color:#308080; ">,</span> first_frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
    <span style="color:#595979; ">#--------------------------------------------</span>
    <span style="color:#595979; "># Step 2: save the first frame for annotation</span>
    <span style="color:#595979; ">#--------------------------------------------</span>
    <span style="color:#595979; "># the first frame vfile name</span>
    output_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">".jpg"</span>
    <span style="color:#595979; "># save the frame</span>
    cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>output_file_path<span style="color:#308080; ">,</span> first_frame<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>

<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># clear the video capture object:</span>
<span style="color:#595979; ">#--------------------------------------------</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># close all windows</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>

### 4.4. Step 4: Manually annotate of the object of interest from the first frame:

#### 4.4.1 Manually annotate of the object of interest from the first frame:

* The bounding-box of the object of interest annotation has the following coordinates:
    * b-box: (tlc_x, tlc_y, width, height) = (445, 100, 76, 183)
      * TLC = (445, 100)
      * TRC = (521, 100)
      * BRC = (521, 283)
      * BLC = (445, 283)


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># The object of interest manually annotated </span>
<span style="color:#595979; "># bounding-box:</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># TLC coordinates: (tlc_x, tlc_y)</span>
<span style="color:#595979; "># tlc-x</span>
tlc_x <span style="color:#308080; ">=</span> <span style="color:#008c00; ">445</span>
<span style="color:#595979; "># tlc-y</span>
tlc_y <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span>
<span style="color:#595979; "># the bounding-box dimension</span>
<span style="color:#595979; "># width</span>
width <span style="color:#308080; ">=</span> <span style="color:#008c00; ">76</span>
<span style="color:#595979; "># height</span>
height <span style="color:#308080; ">=</span> <span style="color:#008c00; ">183</span>
</pre>

#### 4.4.2. Display the annotated bounding box of the object of interest on the first frame:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># Step 1: Read the first frame image</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># the first frame file name</span>
first_frame_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">".jpg"</span>
<span style="color:#595979; "># read the frame image</span>
first_frame <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>imread<span style="color:#308080; ">(</span>output_file_path<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>

<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 2: draw a GREEN rectangle to </span>
<span style="color:#595979; ">#        visualize the bounding rect</span>
<span style="color:#595979; ">#        of the object of interest.</span>
<span style="color:#595979; ">#----------------------------------------</span>
cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>first_frame<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>tlc_x<span style="color:#308080; ">,</span> tlc_y<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>tlc_x <span style="color:#44aadd; ">+</span> width<span style="color:#308080; ">,</span> tlc_y <span style="color:#44aadd; ">+</span> height<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># display the frame image with the </span>
<span style="color:#595979; "># overlaid object of interest </span>
<span style="color:#595979; "># bounding-box</span>
<span style="color:#595979; ">#----------------------------------------</span>
cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Object of interest"</span><span style="color:#308080; ">,</span> first_frame<span style="color:#308080; ">)</span>
<span style="color:#595979; "># wait to visualize the image</span>
cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># close all windows</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># save the figure</span>
<span style="color:#595979; "># the first frame file name</span>
first_frame_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"-bbox-overlaid.jpg"</span>
<span style="color:#595979; "># save the frame</span>
cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>first_frame_file_path<span style="color:#308080; ">,</span> first_frame<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
            
</pre>

<img src="images/frame-0-bbox-overlaid.jpg" width="1000"/>

### 4.5. Step 5: Track the object of interest:

* We are now ready to start tracking the moving object of interest bounding-box using built-in OpenCV Python tracking functionalities:
  * Optical flow
  * Dense optical flow
  * Mean-Shift
  * Cam-Shift
  * BOOSTING Tracker
  * MIL
  * KCF
  * TLD
  * MEDIAN FLOW

#### 4.5.1. Tracker: Lucas Kanade Optical Flow:
In this section, we shall compute Shi-Tomasi corners and track them using the Lucas-Kanade Optical Flow tracker:
From the first frame, we compute compute Shi-Tomasi corners from within the object of interest ROI selected above
Then we apply the Lucas-Kanade Optical Flow algorithm to track these features in the subsequent frames.

##### 4.5.1.1. Set the Shi-Tomasi corner detector parameters:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Parameters for Shi-Tomasi corner detection (good features to track paper)</span>
corner_track_params <span style="color:#308080; ">=</span> <span style="color:#400000; ">dict</span><span style="color:#308080; ">(</span>maxCorners <span style="color:#308080; ">=</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span>
                       qualityLevel <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.3</span><span style="color:#308080; ">,</span>
                       minDistance <span style="color:#308080; ">=</span> <span style="color:#008c00; ">7</span><span style="color:#308080; ">,</span>
                       blockSize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">7</span> <span style="color:#308080; ">)</span>
</pre>

##### 4.5.1.2. Set the Lucas-Kanade optical flow parameters:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Parameters for lucas kanade optical flow</span>
lk_params <span style="color:#308080; ">=</span> <span style="color:#400000; ">dict</span><span style="color:#308080; ">(</span> winSize  <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">200</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">200</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
                  maxLevel <span style="color:#308080; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span>
                  criteria <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_EPS <span style="color:#44aadd; ">|</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_COUNT<span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span><span style="color:#008000; ">0.03</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
</pre>

##### 4.5.1.3. Create a mask to focus on the selected object of interest ROI:

* Anything outside the object of interest bounding-box will be masked-out


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a zero-mask with  the same size as the image</span>
roi_mask <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>img<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> img<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> dtype<span style="color:#308080; ">=</span><span style="color:#400000; ">int</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the pixels inside the object of interest bounding-box to 1</span>
roi_mask<span style="color:#308080; ">[</span>tlc_y<span style="color:#308080; ">:</span> tlc_y <span style="color:#44aadd; ">+</span> height<span style="color:#308080; ">,</span> tlc_x<span style="color:#308080; ">:</span> tlc_x <span style="color:#44aadd; ">+</span> width<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
</pre>


##### 4.5.1.4. Read the first frame and compute features within the object of interest ROI to track:

* We only compute Harris corners inside the object of interest ROI
* We shall then track these features in subsequent frames, using the Optical flow algorithm


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># open the video file</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot read video file: "</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># Grab the very first frame of the stream</span>
ret<span style="color:#308080; ">,</span> prev_frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Convert to grayscale image if it is RGB</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>prev_frame<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">2</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    prev_gray <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>prev_frame<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2GRAY<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
    prev_gray <span style="color:#308080; ">=</span> prev_frame<span style="color:#308080; ">.</span>copy<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Since we only want to compute Harris corners features to track </span>
<span style="color:#595979; "># inside the object of interest ROI:</span>
<span style="color:#595979; ">#  - we need to apply the mask</span>
prev_gray <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>multiply<span style="color:#308080; ">(</span>prev_gray<span style="color:#308080; ">,</span> roi_mask<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># compute the Harris corner</span>
prevPts <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>goodFeaturesToTrack<span style="color:#308080; ">(</span>prev_gray<span style="color:#308080; ">,</span> mask <span style="color:#308080; ">=</span> <span style="color:#074726; ">None</span><span style="color:#308080; ">,</span> <span style="color:#44aadd; ">**</span>corner_track_params<span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Number of detected Harris corners = '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>prevPts<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># Drawing circles around corners</span>
<span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>prevPts<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    temp <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>prevPts<span style="color:#308080; ">[</span>i<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
    cv2<span style="color:#308080; ">.</span>circle<span style="color:#308080; ">(</span>first_frame<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>temp<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> temp<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">5</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># Showing the result</span>
cv2<span style="color:#308080; ">.</span>namedWindow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Detected Harris corners"</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the image with overlaid features</span>
cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Detected Harris corners"</span><span style="color:#308080; ">,</span> first_frame<span style="color:#308080; ">)</span>
<span style="color:#595979; "># save the figure</span>
<span style="color:#595979; "># the first frame file name</span>
first_frame_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-1-harris-corners-overlaid.jpg"</span>
<span style="color:#595979; "># save the frame</span>
cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>first_frame_file_path<span style="color:#308080; ">,</span> first_frame<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># wait to visualize the image</span>
cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># close all windows</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>  


Number of detected Harris corners <span style="color:#308080; ">=</span> <span style="color:#008c00; ">10</span>
</pre>

<img src="images/frame-1-harris-corners-overlaid.jpg" width="1000"/>

##### 4.5.1.5. Track the computed features within the object of interest ROI to track using Lucas-Kanade Optical Flow:

* Next, we start tracking these features in subsequent frames, using the Lucas-Kanade Optical flow algorithm


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979;"># Create a matching mask of the previous frame for drawing on later</span>
mask <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros_like<span style="color:#308080; ">(</span>prev_frame<span style="color:#308080; ">)</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># read the subsequent video frames</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    
    <span style="color:#595979; "># Grab current frame</span>
    ret<span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Grab gray scale</span>
    frame_gray <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2GRAY<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Calculate the Optical Flow on the Gray Scale Frame</span>
    nextPts<span style="color:#308080; ">,</span> status<span style="color:#308080; ">,</span> err <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcOpticalFlowPyrLK<span style="color:#308080; ">(</span>prev_gray<span style="color:#308080; ">,</span> frame_gray<span style="color:#308080; ">,</span> prevPts<span style="color:#308080; ">,</span> <span style="color:#074726; ">None</span><span style="color:#308080; ">,</span> <span style="color:#44aadd; ">**</span>lk_params<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Using the returned status array (the status output)</span>
    <span style="color:#595979; "># status output status vector (of unsigned chars); each element of the vector is set to 1 if</span>
    <span style="color:#595979; "># the flow for the corresponding features has been found, otherwise, it is set to 0.</span>
    good_new <span style="color:#308080; ">=</span> nextPts<span style="color:#308080; ">[</span>status<span style="color:#44aadd; ">==</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
    good_prev <span style="color:#308080; ">=</span> prevPts<span style="color:#308080; ">[</span>status<span style="color:#44aadd; ">==</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
    
    <span style="color:#595979; "># Use ravel to get points to draw lines and circles</span>
    <span style="color:#200080; font-weight:bold; ">for</span> i<span style="color:#308080; ">,</span><span style="color:#308080; ">(</span>new<span style="color:#308080; ">,</span>prev<span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">enumerate</span><span style="color:#308080; ">(</span><span style="color:#400000; ">zip</span><span style="color:#308080; ">(</span>good_new<span style="color:#308080; ">,</span>good_prev<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        
        x_new<span style="color:#308080; ">,</span>y_new <span style="color:#308080; ">=</span> new<span style="color:#308080; ">.</span>ravel<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        x_prev<span style="color:#308080; ">,</span>y_prev <span style="color:#308080; ">=</span> prev<span style="color:#308080; ">.</span>ravel<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Lines will be drawn using the mask created from the first frame</span>
        mask <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>line<span style="color:#308080; ">(</span>mask<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x_new<span style="color:#308080; ">,</span>y_new<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#308080; ">(</span>x_prev<span style="color:#308080; ">,</span>y_prev<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Draw red circles at corner points</span>
        frame <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>circle<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span><span style="color:#308080; ">(</span>x_new<span style="color:#308080; ">,</span>y_new<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; ">#----------------------------------</span>
    <span style="color:#595979; "># Display the tracking results:</span>
    <span style="color:#595979; ">#----------------------------------</span>
    <span style="color:#595979; "># - only do this for some frames</span>
    <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> frame_counter <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># add the frame with overlays to the mask </span>
        img <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>add<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span>mask<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># display the image with overlays</span>
        cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Tracker: Lucas-Kanade Optical Flow on frame #: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>img<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># save the figure</span>
        <span style="color:#595979; "># the first frame file name</span>
        output_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"-optical-flow-tracker.jpg"</span>
        <span style="color:#595979; "># save the frame</span>
        cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>output_file_path<span style="color:#308080; ">,</span> img<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
    <span style="color:#595979; "># increment the frame counter</span>
    frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
    
    <span style="color:#595979; "># quit if user hits: ESC</span>
    k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">30</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
    <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">break</span>
   
    <span style="color:#595979; "># Now update the previous frame and previous points</span>
    prev_gray <span style="color:#308080; ">=</span> frame_gray<span style="color:#308080; ">.</span>copy<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    prevPts <span style="color:#308080; ">=</span> good_new<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
    
    
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

* The figures below illustrates the Lucas-Kanade Optical Flow tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/LK--Optical-Flow--Tracker.png" width="1000"/>

#### 4.5.2. Tracker: Dense Optical Flow:

In this section, we shall implement the Dense Optical Flow tracker in OpenCV Python:
  * Dense optical flow attempts to compute the optical flow vector for every pixel of each frame.
  * While such computation may be slower, it gives a more accurate result and a denser result suitable for applications such as learning structure from motion and video segmentation.
  * Since Dense Optical Flow tracks every pixel, it does not require an intial set of points or region to track
  * Dense optical flow highlights the pixels that are moving/changing faster than the rest of the image.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># open the video file</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot read video file: "</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
<span style="color:#595979; "># Grab first frame    </span>
ret<span style="color:#308080; ">,</span> frame1 <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Get gray scale image of first frame and make a mask in HSV color</span>
prvsImg <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame1<span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>COLOR_BGR2GRAY<span style="color:#308080; ">)</span>

hsv_mask <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros_like<span style="color:#308080; ">(</span>frame1<span style="color:#308080; ">)</span>
hsv_mask<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">255</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    ret<span style="color:#308080; ">,</span> frame2 <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    nextImg <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame2<span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>COLOR_BGR2GRAY<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Check out the markdown text above for a break down of these paramters, most of these are just suggested defaults</span>
    flow <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcOpticalFlowFarneback<span style="color:#308080; ">(</span>prvsImg<span style="color:#308080; ">,</span>nextImg<span style="color:#308080; ">,</span> <span style="color:#074726; ">None</span><span style="color:#308080; ">,</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">15</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">5</span><span style="color:#308080; ">,</span> <span style="color:#008000; ">1.2</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
    
    
    <span style="color:#595979; "># Color the channels based on the angle of travel</span>
    <span style="color:#595979; "># Pay close attention to your video, the path of the direction of flow will determine color!</span>
    mag<span style="color:#308080; ">,</span> ang <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cartToPolar<span style="color:#308080; ">(</span>flow<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> flow<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span>angleInDegrees<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    hsv_mask<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> ang<span style="color:#44aadd; ">/</span><span style="color:#008c00; ">2</span>
    hsv_mask<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>normalize<span style="color:#308080; ">(</span>mag<span style="color:#308080; ">,</span><span style="color:#074726; ">None</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>NORM_MINMAX<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; ">#----------------------------------</span>
    <span style="color:#595979; "># Display the tracking results:</span>
    <span style="color:#595979; ">#----------------------------------</span>
    <span style="color:#595979; "># - only do this for some frames</span>
    <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> frame_counter <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># Convert back to BGR to show with imshow from cv</span>
        bgr <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>hsv_mask<span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>COLOR_HSV2BGR<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># display the color image </span>
        cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Tracker: Dense Optical Flow on frame #: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>bgr<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># save the figure</span>
        <span style="color:#595979; "># the first frame file name</span>
        output_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"-dense-optical-flow-tracker.jpg"</span>
        <span style="color:#595979; "># save the frame</span>
        cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>output_file_path<span style="color:#308080; ">,</span> bgr<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
    <span style="color:#595979; "># increment the frame counter</span>
    frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
    
    <span style="color:#595979; "># quit if user hits: ESC</span>
    k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">30</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
    <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">break</span>
    
    <span style="color:#595979; "># Set the Previous image as the next iamge for the loop</span>
    prvsImg <span style="color:#308080; ">=</span> nextImg

    
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

* The figures below illustrates the Dense Optical Flow tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/Dense--Optical-Flow--Tracker.png" width="1000"/>

#### 4.5.3. Tracker: Mean-Shift Tracker:

* In this section, we shall implement the Mean-Shift Tracker in OpenCV Python:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># open the video file</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot read video file: "</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
<span style="color:#595979; "># take first frame of the video</span>
ret<span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Set Up the Initial Tracking Window:</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># - This is set to the manually selected object of interest </span>
<span style="color:#595979; ">#   ROI/bounding-box previously selected</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># The object of interest manually annotated </span>
<span style="color:#595979; "># bounding-box:</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># TLC coordinates: (tlc_x, tlc_y)</span>
<span style="color:#595979; "># tlc-x</span>
<span style="color:#595979; "># tlc_x = 445</span>
<span style="color:#595979; "># tlc-y</span>
<span style="color:#595979; "># tlc_y = 100</span>
<span style="color:#595979; "># the bounding-box dimension</span>
<span style="color:#595979; "># width</span>
<span style="color:#595979; "># width = 76</span>
<span style="color:#595979; "># height</span>
<span style="color:#595979; "># height = 183</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># setup the tracked window bounding-box in </span>
<span style="color:#595979; "># the form: (x,y,w,h)</span>
<span style="color:#595979; ">#--------------------------------------------</span>
track_window <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>tlc_x<span style="color:#308080; ">,</span> tlc_y<span style="color:#308080; ">,</span> width<span style="color:#308080; ">,</span> height<span style="color:#308080; ">)</span>
<span style="color:#595979; "># set up the ROI for tracking</span>
roi <span style="color:#308080; ">=</span> frame<span style="color:#308080; ">[</span>tlc_y<span style="color:#308080; ">:</span>tlc_y<span style="color:#44aadd; ">+</span>height<span style="color:#308080; ">,</span> tlc_x<span style="color:#308080; ">:</span>tlc_x<span style="color:#44aadd; ">+</span>width<span style="color:#308080; ">]</span>

<span style="color:#595979; "># Use the HSV Color Mapping</span>
hsv_roi <span style="color:#308080; ">=</span>  cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>roi<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Find histogram to backproject the target on each frame for calculation of meanshit</span>
roi_hist <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcHist<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv_roi<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#074726; ">None</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Normalize the histogram array values given a min of 0 and max of 255</span>
cv2<span style="color:#308080; ">.</span>normalize<span style="color:#308080; ">(</span>roi_hist<span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>NORM_MINMAX<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Setup the termination criteria, either 10 iteration or move by at least 1 pt</span>
term_crit <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_EPS <span style="color:#44aadd; ">|</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_COUNT<span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># iterate over the farmes</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    ret <span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> ret <span style="color:#44aadd; ">==</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
        
        <span style="color:#595979; "># Grab the Frame in HSV</span>
        hsv <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Calculate the Back Projection based off the roi_hist created earlier</span>
        dst <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcBackProject<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Apply meanshift to get the new coordinates of the rectangle</span>
        ret<span style="color:#308080; ">,</span> track_window <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>meanShift<span style="color:#308080; ">(</span>dst<span style="color:#308080; ">,</span> track_window<span style="color:#308080; ">,</span> term_crit<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># Display the tracking results:</span>
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># - only do this for some frames</span>
        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> frame_counter <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>  
            <span style="color:#595979; "># Draw the new rectangle on the image</span>
            x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h <span style="color:#308080; ">=</span> track_window
            img2 <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">,</span>y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">)</span>
            <span style="color:#595979; "># display the image</span>
            cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Tracker: Mean-Shift on frame #: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>img2<span style="color:#308080; ">)</span>
            <span style="color:#595979; "># save the figure</span>
            <span style="color:#595979; "># the first frame file name</span>
            output_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"-mean-shift-tracker.jpg"</span>
            <span style="color:#595979; "># save the frame</span>
            cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>output_file_path<span style="color:#308080; ">,</span> img2<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>

        <span style="color:#595979; "># increment the frame counter</span>
        frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
        
        <span style="color:#595979; "># quit if user hits: ESC</span>
        k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
        <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span>
            <span style="color:#200080; font-weight:bold; ">break</span>
        
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">break</span>
        
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

* The figures below illustrates the Mean-Shift flow tracking of the points of interest for frame 100, 200, 300 and 400, respectively.


<img src="images/Mean-Shift-Tracker.png" width="1000"/>

#### 4.5.4.  Tracker: Cam-Shift Tracker:

In this section, we shall implement the Cam-Shift Tracker in OpenCV Python:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># open the video file</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot read video file: "</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># take first frame of the video</span>
ret<span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Set Up the Initial Tracking Window:</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># - This is set to the manually selected object of interest </span>
<span style="color:#595979; ">#   ROI/bounding-box previously selected</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># The object of interest manually annotated </span>
<span style="color:#595979; "># bounding-box:</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># TLC coordinates: (tlc_x, tlc_y)</span>
<span style="color:#595979; "># tlc-x</span>
<span style="color:#595979; "># tlc_x = 445</span>
<span style="color:#595979; "># tlc-y</span>
<span style="color:#595979; "># tlc_y = 100</span>
<span style="color:#595979; "># the bounding-box dimension</span>
<span style="color:#595979; "># width</span>
<span style="color:#595979; "># width = 76</span>
<span style="color:#595979; "># height</span>
<span style="color:#595979; "># height = 183</span>
<span style="color:#595979; ">#--------------------------------------------</span>
<span style="color:#595979; "># setup the tracked window bounding-box in </span>
<span style="color:#595979; "># the form: (x,y,w,h)</span>
<span style="color:#595979; ">#--------------------------------------------</span>
track_window <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>tlc_x<span style="color:#308080; ">,</span> tlc_y<span style="color:#308080; ">,</span> width<span style="color:#308080; ">,</span> height<span style="color:#308080; ">)</span>
<span style="color:#595979; "># set up the ROI for tracking</span>
roi <span style="color:#308080; ">=</span> frame<span style="color:#308080; ">[</span>tlc_y<span style="color:#308080; ">:</span>tlc_y<span style="color:#44aadd; ">+</span>height<span style="color:#308080; ">,</span> tlc_x<span style="color:#308080; ">:</span>tlc_x<span style="color:#44aadd; ">+</span>width<span style="color:#308080; ">]</span>

<span style="color:#595979; "># Use the HSV Color Mapping</span>
hsv_roi <span style="color:#308080; ">=</span>  cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>roi<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Find histogram to backproject the target on each frame for calculation of meanshit</span>
roi_hist <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcHist<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv_roi<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#074726; ">None</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Normalize the histogram array values given a min of 0 and max of 255</span>
cv2<span style="color:#308080; ">.</span>normalize<span style="color:#308080; ">(</span>roi_hist<span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>NORM_MINMAX<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Setup the termination criteria, either 10 iteration or move by at least 1 pt</span>
term_crit <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_EPS <span style="color:#44aadd; ">|</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_COUNT<span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># iterate over the farmes</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    ret <span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> ret <span style="color:#44aadd; ">==</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
        
        <span style="color:#595979; "># Grab the Frame in HSV</span>
        hsv <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Calculate the Back Projection based off the roi_hist created earlier</span>
        dst <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcBackProject<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Apply Camshift to get the new coordinates of the rectangle</span>
        ret<span style="color:#308080; ">,</span> track_window <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>CamShift<span style="color:#308080; ">(</span>dst<span style="color:#308080; ">,</span> track_window<span style="color:#308080; ">,</span> term_crit<span style="color:#308080; ">)</span>
       
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># Display the tracking results:</span>
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># - only do this for some frames</span>
        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> frame_counter <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>  
            <span style="color:#595979; "># Draw the new rectangle on the image</span>
            pts <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>boxPoints<span style="color:#308080; ">(</span>ret<span style="color:#308080; ">)</span>
            pts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>int0<span style="color:#308080; ">(</span>pts<span style="color:#308080; ">)</span>
            img2 <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>polylines<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span><span style="color:#308080; ">[</span>pts<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#074726; ">True</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">)</span>
            cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'img2'</span><span style="color:#308080; ">,</span>img2<span style="color:#308080; ">)</span>
            <span style="color:#595979; "># display the image</span>
            cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Tracker: Cam-Shift on frame #: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>img2<span style="color:#308080; ">)</span>
            <span style="color:#595979; "># save the figure</span>
            <span style="color:#595979; "># the first frame file name</span>
            output_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"-cam-shift-tracker.jpg"</span>
            <span style="color:#595979; "># save the frame</span>
            cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>output_file_path<span style="color:#308080; ">,</span> img2<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
        
        <span style="color:#595979; "># increment the frame counter</span>
        frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
        
        <span style="color:#595979; "># quit if user hits: ESC</span>
        k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
        <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span>
            <span style="color:#200080; font-weight:bold; ">break</span>
        
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">break</span>
        
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

* The figures below illustrates the Cam-Shift tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/Cam-Shift-Tracker.png" width="1000"/>


#### 4.5.5. Additional Object Tracking API:

* In this section, we shall implement the Tracking APIs (Built-in with OpenCV):

  * We get the following options to experiment with the following trackers:
    * Enter 0 for BOOSTING
    * Enter 1 for MIL
    * Enter 2 for KCF
    * Enter 3 for TLD
    * Enter 4 for MEDIANFLOW


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">'''</span>
<span style="color:#595979; ">Gets the user tracker selection:</span>
<span style="color:#595979; ">'''</span>
<span style="color:#200080; font-weight:bold; ">def</span> ask_for_tracker<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Please select the Tracker API would you like to use:"</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 0 for BOOSTING: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 1 for MIL: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 2 for KCF: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 3 for TLD: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 4 for MEDIANFLOW: "</span><span style="color:#308080; ">)</span>
    choice <span style="color:#308080; ">=</span> <span style="color:#400000; ">input</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Please select your tracker: "</span><span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'0'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerBoosting_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'1'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerMIL_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'2'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerKCF_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'3'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerTLD_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'4'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerMedianFlow_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>


    <span style="color:#200080; font-weight:bold; ">return</span> tracker


<span style="color:#595979; "># get the Tracker option from the user</span>
tracker <span style="color:#308080; ">=</span> ask_for_tracker<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the selected tracker</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"User selected Tracker: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>tracker<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>split<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>

Please select the Tracker API would you like to use:
Enter <span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">for</span> BOOSTING<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">1</span> <span style="color:#200080; font-weight:bold; ">for</span> MIL<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">2</span> <span style="color:#200080; font-weight:bold; ">for</span> KCF<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">3</span> <span style="color:#200080; font-weight:bold; ">for</span> TLD<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">4</span> <span style="color:#200080; font-weight:bold; ">for</span> MEDIANFLOW<span style="color:#308080; ">:</span> 
Please select your tracker<span style="color:#308080; ">:</span> <span style="color:#008c00; ">4</span>
User selected Tracker<span style="color:#308080; ">:</span> TrackerMedianFlow

</pre>

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Get the tracker option from the user</span>
tracker <span style="color:#308080; ">=</span> ask_for_tracker<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># get the tracker name</span>
tracker_name <span style="color:#308080; ">=</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>tracker<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>split<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span>

<span style="color:#595979; "># open the video file</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span>video_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot read video file: "</span> <span style="color:#44aadd; ">+</span> video_file_path<span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># take first frame of the video</span>
ret<span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Special function allows us to draw on the very first frame our desired ROI</span>
roi <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>selectROI<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> <span style="color:#074726; ">False</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Initialize tracker with first frame and bounding box</span>
ret <span style="color:#308080; ">=</span> tracker<span style="color:#308080; ">.</span>init<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> roi<span style="color:#308080; ">)</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># iterate over the farmes</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; "># Read a new frame</span>
    ret<span style="color:#308080; ">,</span> frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    
    
    <span style="color:#595979; "># Update tracker</span>
    success<span style="color:#308080; ">,</span> roi <span style="color:#308080; ">=</span> tracker<span style="color:#308080; ">.</span>update<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># roi variable is a tuple of 4 floats</span>
    <span style="color:#595979; "># We need each value and we need them as integers</span>
    <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> <span style="color:#400000; ">tuple</span><span style="color:#308080; ">(</span><span style="color:#400000; ">map</span><span style="color:#308080; ">(</span><span style="color:#400000; ">int</span><span style="color:#308080; ">,</span>roi<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Draw Rectangle as Tracker moves</span>
    <span style="color:#200080; font-weight:bold; ">if</span> success<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># Tracking success</span>
        p1 <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span> y<span style="color:#308080; ">)</span>
        p2 <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">,</span> y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">)</span>
        cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> p1<span style="color:#308080; ">,</span> p2<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">else</span> <span style="color:#308080; ">:</span>
        <span style="color:#595979; "># Tracking failure</span>
        cv2<span style="color:#308080; ">.</span>putText<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">"Failure to Detect Tracking!!"</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">100</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">200</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>FONT_HERSHEY_SIMPLEX<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># Display tracker type on frame</span>
    cv2<span style="color:#308080; ">.</span>putText<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> tracker_name<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">20</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">400</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>FONT_HERSHEY_SIMPLEX<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>

    <span style="color:#595979; ">#----------------------------------</span>
    <span style="color:#595979; "># Display the tracking results:</span>
    <span style="color:#595979; ">#----------------------------------</span>
    <span style="color:#595979; "># - only do this for some frames</span>
    <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> frame_counter <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>  
        <span style="color:#595979; "># Draw the new rectangle on the image</span>
        cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Tracker: '</span> <span style="color:#44aadd; ">+</span> tracker_name <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">'on frame #: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> frame<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># save the figure</span>
        <span style="color:#595979; "># the first frame file name</span>
        output_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../results/frame-"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>frame_counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">'--'</span> <span style="color:#44aadd; ">+</span> tracker_name <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">".jpg"</span>
        <span style="color:#595979; "># save the frame</span>
        cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span>output_file_path<span style="color:#308080; ">,</span> frame<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
        
    <span style="color:#595979; "># increment the frame counter</span>
    frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
        
    <span style="color:#595979; "># Exit if ESC pressed</span>
    k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
    <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span> <span style="color:#308080; ">:</span> 
        <span style="color:#200080; font-weight:bold; ">break</span>
        
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>


Please select the Tracker API would you like to use:
Enter <span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">for</span> BOOSTING<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">1</span> <span style="color:#200080; font-weight:bold; ">for</span> MIL<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">2</span> <span style="color:#200080; font-weight:bold; ">for</span> KCF<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">3</span> <span style="color:#200080; font-weight:bold; ">for</span> TLD<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">4</span> <span style="color:#200080; font-weight:bold; ">for</span> MEDIANFLOW<span style="color:#308080; ">:</span> 
Please select your tracker<span style="color:#308080; ">:</span> <span style="color:#008c00; ">4</span>
</pre>

##### 4.5.5.1. Object Tracking API: BOOSTING Tracker

* Enter 0 for BOOSTING
* The figures below illustrates the Boosting tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/Boosting-Tracker.png" width="1000"/>

##### 4.5.5.2. Object Tracking API: MIL Tracker

* Enter 1 for MIL
* The figures below illustrates the MIL tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/MIL-Tracker.png" width="1000"/>

##### 4.5.5.2. Object Tracking API: KCF Tracker

* Enter 2 for KCF
* The figures below illustrates the KCF  tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/KCF-Tracker.png" width="1000"/>

##### 4.5.5.3. Object Tracking API: TLD Tracker

* Enter 3 for TLD
* The figures below illustrates the TLD tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/LTD-Tracker.png" width="1000"/>

##### 4.5.5.4. Object Tracking API: MEDIAN FLOW Tracker

* Enter 4 for MEDIANFLOW
* The figures below illustrates the Median-Flow tracking of the points of interest for frame 100, 200, 300 and 400, respectively.

<img src="images/Median-Flow-Tracker.png" width="1000"/>


## 5. Analysis

* We have demonstrated tracking a moving object of interest using 9 tracking algorithms implemented in OpenCV Python:

  * The 5 tracking algorithms implemented in the OpenCV Tracking API all performed equally and extremely well, yielding nearly perfect tracking of the moving object of interest.
    * BOOSTING
    * MIL
    * KCF
    * TLD
    * MEDIAN FLOW
  * The other 4 tracking algorithms generally yield poor tracking results:
    * Optical flow
    * Dense optical flow
    * Mean-Shift
    * Cam-Shift


## 6. Future Work

* We propose to explore the following related issues:

  * To explore these implemented tracking algorithm and get a better understating of:
    * How each algorithm works
    * The advantages and limitations of each algorithm
    * To implement multi-object trackers for tracking and distinguishing multiple objects at the same time.


## 7. References

1. Adrian Rosebrock. OpenCV Object Tracking. https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/ 
2. Adrian Rosebrock. Simple object tracking with OpenCV. https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/ 
3. Satya Mallick. Object Tracking using OpenCV. https://learnopencv.com/object-tracking-using-opencv-cpp-python/ 
4. Anna Petrovicheva. Multiple Object Tracking in Realtime. https://opencv.org/multiple-object-tracking-in-realtime/ 
5. Ehsan Gazar. Object Tracking with OpenCV. https://ehsangazar.com/object-tracking-with-opencv-fd18ccdd7369 
6. Automatic Addison. Real-Time Object Tracking Using OpenCV and a Webcam. https://automaticaddison.com/real-time-object-tracking-using-opencv-and-a-webcam/ 
7. Automatic Addison. How to Do Multiple Object Tracking Using OpenCV. https://automaticaddison.com/how-to-do-multiple-object-tracking-using-opencv/
 
 
