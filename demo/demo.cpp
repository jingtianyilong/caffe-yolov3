// #include <stdio.h>
// #include <signal.h>
// #include <unistd.h>
// #include <sys/time.h>

// #include "detector.h"

// using namespace cv;

// const char *labels[80] = {
// "person",
// "bicycle",
// "car",
// "motorbike",
// "aeroplane",
// "bus",
// "train",
// "truck",
// "boat",
// "traffic light",
// "fire hydrant",
// "stop sign",
// "parking meter",
// "bench",
// "bird",
// "cat",
// "dog",
// "horse",
// "sheep",
// "cow",
// "elephant",
// "bear",
// "zebra",
// "giraffe",
// "backpack",
// "umbrella",
// "handbag",
// "tie",
// "suitcase",
// "frisbee",
// "skis",
// "snowboard",
// "sports ball",
// "kite",
// "baseball bat",
// "baseball glove",
// "skateboard",
// "surfboard",
// "tennis racket",
// "bottle",
// "wine glass",
// "cup",
// "fork",
// "knife",
// "spoon",
// "bowl",
// "banana",
// "apple",
// "sandwich",
// "orange",
// "broccoli",
// "carrot",
// "hot dog",
// "pizza",
// "donut",
// "cake",
// "chair",
// "sofa",
// "pottedplant",
// "bed",
// "diningtable",
// "toilet",
// "tvmonitor",
// "laptop",
// "mouse",
// "remote",
// "keyboard",
// "cell phone",
// "microwave",
// "oven",
// "toaster",
// "sink",
// "refrigerator",
// "book",
// "clock",
// "vase",
// "scissors",
// "teddy bear",
// "hair drier",
// "toothbrush"
// };

// bool signal_recieved = false;


// void sig_handler(int signo){
//     if( signo == SIGINT ){
//             printf("received SIGINT\n");
//             signal_recieved = true;
//     }
// }
// float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

// float get_color(int c, int x, int max)
// {
//     float ratio = ((float)x/max)*5;
//     int i = floor(ratio);
//     int j = ceil(ratio);
//     ratio -= i;
//     float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
//     return r;
// }

// uint64_t current_timestamp() {
//     struct timeval te; 
//     gettimeofday(&te, NULL); // get current time
//     return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
// }

// int main( int argc, char** argv )
// {
//     std::string model_file;
//     std::string weights_file;
//     int conf_thres;
//     if(4 == argc){
//         model_file = argv[1];
//         weights_file = argv[2];
//         conf_thres = std::stoi(argv[3]);
//     }
//     else{
//         LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [conf_thres]";
//         return -1;
//     }	
//     int gpu_id = 0;
//     //init network
//     Detector detector = Detector(model_file,weights_file,gpu_id);

//     //load image with opencv
//     int deviceID = 0;             // 0 = open default camera
//     VideoCapture cap;
//     cap.open(deviceID);
//     if (!cap.isOpened()){
//         std::cout << "ERROR! Unable to open camera\n";
//         return -1;
//     }

//     Mat img;
    
//     //detect
//     for (;;)
//     {
//         cap.read(img);
//         if (img.empty()) {break;}
//         float thresh = 0.3;
//         std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);
//         int classes = 80;
//         int width = std::max(1.0f, img.rows * .005f);
//         //show detection results
//         for (int i=0;i<bbox_vec.size();++i){
//             bbox_t b = bbox_vec[i];
//             if (b.prob >= conf_thres){
//                 int offset = b.obj_id * 123457 % classes;
//                 float red = get_color(2, offset, classes);
//                 float green = get_color(1, offset, classes);
//                 float blue = get_color(0, offset, classes);
//                 float rgb[3];
//                 rgb[0] = red;
//                 rgb[1] = green;
//                 rgb[2] = blue;
//                 float const font_size = img.rows / 700.F;
//                 std::stringstream ss;
//                 ss << labels[b.obj_id] << " " << std::setprecision(2) << b.prob;
//                 std::string labelstr = ss.str();
//                 cv::Size const text_size = cv::getTextSize(labelstr, cv::FONT_HERSHEY_SIMPLEX, font_size, 1, 0);
//                 cv::Point pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
//                 pt1.x  = b.x;
//                 pt2.x  = b.x + b.w;
//                 pt1.y  = b.y;
//                 pt2.y  = b.y + b.h;
//                 pt_text.x = b.x;
//                 pt_text.y = b.y - 4;// 12;
//                 pt_text_bg1.x = b.x;
//                 pt_text_bg1.y = b.y - (3 + 18 * font_size);
//                 pt_text_bg2.x = b.x + text_size.width;
//                 pt_text_bg2.y = pt1.y;
//                 cv::Scalar color;
//                 color.val[0] = red * 256;
//                 color.val[1] = green * 256;
//                 color.val[2] = blue * 256;
//                 rectangle(img,pt1,pt2,color,width,8,0);
//                 rectangle(img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);    // filled
//                 cv::Scalar white_color = CV_RGB(255, 255, 255);
//                 putText(img, labelstr, pt_text, cv::FONT_HERSHEY_SIMPLEX, font_size, white_color, 2 * font_size, CV_AA);
//             }
            
//         }
//         imshow("show",img);
//         if (waitKey (5)>= 0){break;}
//     }
//     return 0;
// }

#include <ros/ros.h>
#include <signal.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "detector.h"

const char *labels[80] = {
"person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
};
int gpu_id = 0;
//init network
std::string model_file="/apollo/caffe-yolov4/prototxt/yolov4.prototxt";
std::string weights_file="/apollo/caffe-yolov4/yolov4.caffemodel";
Detector detector = Detector(model_file,weights_file,gpu_id);bool signal_recieved = false;

void sigHandler(int sig){
  ros::shutdown();
}

void sig_handler(int signo){
    if( signo == SIGINT ){
            printf("received SIGINT\n");
            signal_recieved = true;
    }
}
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

const unsigned char uchar_clipping_table[] = {
    0,   0,   0,   0,   0,   0,   0,
    0,  // -128 - -121
    0,   0,   0,   0,   0,   0,   0,
    0,  // -120 - -113
    0,   0,   0,   0,   0,   0,   0,
    0,  // -112 - -105
    0,   0,   0,   0,   0,   0,   0,
    0,  // -104 -  -97
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -96 -  -89
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -88 -  -81
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -80 -  -73
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -72 -  -65
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -64 -  -57
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -56 -  -49
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -48 -  -41
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -40 -  -33
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -32 -  -25
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -24 -  -17
    0,   0,   0,   0,   0,   0,   0,
    0,  //  -16 -   -9
    0,   0,   0,   0,   0,   0,   0,
    0,  //   -8 -   -1
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
    30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
    45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
    60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
    75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
    90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
    255, 255, 255, 255, 255, 255, 255, 255, 255,  // 256-263
    255, 255, 255, 255, 255, 255, 255, 255,       // 264-271
    255, 255, 255, 255, 255, 255, 255, 255,       // 272-279
    255, 255, 255, 255, 255, 255, 255, 255,       // 280-287
    255, 255, 255, 255, 255, 255, 255, 255,       // 288-295
    255, 255, 255, 255, 255, 255, 255, 255,       // 296-303
    255, 255, 255, 255, 255, 255, 255, 255,       // 304-311
    255, 255, 255, 255, 255, 255, 255, 255,       // 312-319
    255, 255, 255, 255, 255, 255, 255, 255,       // 320-327
    255, 255, 255, 255, 255, 255, 255, 255,       // 328-335
    255, 255, 255, 255, 255, 255, 255, 255,       // 336-343
    255, 255, 255, 255, 255, 255, 255, 255,       // 344-351
    255, 255, 255, 255, 255, 255, 255, 255,       // 352-359
    255, 255, 255, 255, 255, 255, 255, 255,       // 360-367
    255, 255, 255, 255, 255, 255, 255, 255,       // 368-375
    255, 255, 255, 255, 255, 255, 255, 255,       // 376-383
};
const int clipping_table_offset = 128;

/** Clip a value to the range 0<val<255. For speed this is done using an
 * array, so can only cope with numbers in the range -128<val<383.
 */
static unsigned char CLIPVALUE(int val) {
  // Old method (if)
  /*   val = val < 0 ? 0 : val; */
  /*   return val > 255 ? 255 : val; */

  // New method (array)
  return uchar_clipping_table[val + clipping_table_offset];
}

/**
 * Conversion from YUV to RGB.
 * The normal conversion matrix is due to Julien (surname unknown):
 *
 * [ R ]   [  1.0   0.0     1.403 ] [ Y ]
 * [ G ] = [  1.0  -0.344  -0.714 ] [ U ]
 * [ B ]   [  1.0   1.770   0.0   ] [ V ]
 *
 * and the firewire one is similar:
 *
 * [ R ]   [  1.0   0.0     0.700 ] [ Y ]
 * [ G ] = [  1.0  -0.198  -0.291 ] [ U ]
 * [ B ]   [  1.0   1.015   0.0   ] [ V ]
 *
 * Corrected by BJT (coriander's transforms RGB->YUV and YUV->RGB
 *                   do not get you back to the same RGB!)
 * [ R ]   [  1.0   0.0     1.136 ] [ Y ]
 * [ G ] = [  1.0  -0.396  -0.578 ] [ U ]
 * [ B ]   [  1.0   2.041   0.002 ] [ V ]
 *
 */

static void YUV2RGB(const unsigned char y, const unsigned char u,
                    const unsigned char v, unsigned char *r, unsigned char *g,
                    unsigned char *b) {
  const int y2 = (int)y;
  const int u2 = (int)u - 128;
  const int v2 = (int)v - 128;
  // std::cerr << "YUV=("<<y2<<","<<u2<<","<<v2<<")"<<std::endl;

  // This is the normal YUV conversion, but
  // appears to be incorrect for the firewire cameras
  //   int r2 = y2 + ( (v2*91947) >> 16);
  //   int g2 = y2 - ( ((u2*22544) + (v2*46793)) >> 16 );
  //   int b2 = y2 + ( (u2*115999) >> 16);
  // This is an adjusted version (UV spread out a bit)
  int r2 = y2 + ((v2 * 37221) >> 15);
  int g2 = y2 - (((u2 * 12975) + (v2 * 18949)) >> 15);
  int b2 = y2 + ((u2 * 66883) >> 15);
  // std::cerr << "   RGB=("<<r2<<","<<g2<<","<<b2<<")"<<std::endl;

  // Cap the values.
  *r = CLIPVALUE(r2);
  *g = CLIPVALUE(g2);
  *b = CLIPVALUE(b2);
}

static void yuyv2bgr(unsigned char *YUV, unsigned char *RGB, int NumPixels) {
  int i, j;
  unsigned char y0, y1, u, v;
  unsigned char r, g, b;

  for (i = 0, j = 0; i < (NumPixels << 1); i += 4, j += 6) {
    y0 = (unsigned char)YUV[i + 0];
    u = (unsigned char)YUV[i + 1];
    y1 = (unsigned char)YUV[i + 2];
    v = (unsigned char)YUV[i + 3];
    YUV2RGB(y0, u, v, &r, &g, &b);
    RGB[j + 0] = b;
    RGB[j + 1] = g;
    RGB[j + 2] = r;
    YUV2RGB(y1, u, v, &r, &g, &b);
    RGB[j + 3] = b;
    RGB[j + 4] = g;
    RGB[j + 5] = r;
  }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::Mat img;
    img = cv::Mat(msg->height, msg->width, CV_8UC3);
    int pixel_num = msg->width * msg->height;
    unsigned char *yuv = (unsigned char *)&(msg->data[0]);
    yuyv2bgr(yuv, img.data, pixel_num);    
    float thresh = 0.3;
    std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);
    int classes = 80;
    int width = std::max(1.0f, img.rows * .005f);
    //show detection results
    for (int i=0;i<bbox_vec.size();++i){
        bbox_t b = bbox_vec[i];
        if (b.prob >= 0.45){
            int offset = b.obj_id * 123457 % classes;
            float red = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue = get_color(0, offset, classes);
            float rgb[3];
            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            float const font_size = img.rows / 700.F;
            std::stringstream ss;
            ss << labels[b.obj_id] << " " << std::setprecision(2) << b.prob;
            std::string labelstr = ss.str();
            cv::Size const text_size = cv::getTextSize(labelstr, cv::FONT_HERSHEY_SIMPLEX, font_size, 1, 0);
            cv::Point pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
            pt1.x  = b.x;
            pt2.x  = b.x + b.w;
            pt1.y  = b.y;
            pt2.y  = b.y + b.h;
            pt_text.x = b.x;
            pt_text.y = b.y - 4;// 12;
            pt_text_bg1.x = b.x;
            pt_text_bg1.y = b.y - (3 + 18 * font_size);
            pt_text_bg2.x = b.x + text_size.width;
            pt_text_bg2.y = pt1.y;
            cv::Scalar color;
            color.val[0] = red * 256;
            color.val[1] = green * 256;
            color.val[2] = blue * 256;
            rectangle(img,pt1,pt2,color,width,8,0);
            rectangle(img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);    // filled
            cv::Scalar white_color = CV_RGB(255, 255, 255);
            putText(img, labelstr, pt_text, cv::FONT_HERSHEY_SIMPLEX, font_size, white_color, 2 * font_size, CV_AA);
        }
        
    }
    cv::imshow("view", img);
    int c = cv::waitKey(1);
    if ((char)c == 27) {    
      cv::destroyWindow("view");
      signal(SIGINT,sigHandler);
}
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view",cv::WINDOW_AUTOSIZE);
  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/apollo/sensor/camera/obstacle/front_6mm", 1, imageCallback);
  ros::spin();
  cv::destroyAllWindows();
  return 0;
}