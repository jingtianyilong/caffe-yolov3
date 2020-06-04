#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>

#include "detector.h"

using namespace cv;

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

bool signal_recieved = false;


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

int main( int argc, char** argv )
{
    std::string model_file;
    std::string weights_file;
    int conf_thres;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        conf_thres = std::stoi(argv[3]);
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [conf_thres]";
        return -1;
    }	
    int gpu_id = 0;
    //init network
    Detector detector = Detector(model_file,weights_file,gpu_id);

    //load image with opencv
    int deviceID = 0;             // 0 = open default camera
    VideoCapture cap;
    cap.open(deviceID);
    if (!cap.isOpened()){
        std::cout << "ERROR! Unable to open camera\n";
        return -1;
    }

    Mat img;
    
    //detect
    for (;;)
    {
        cap.read(img);
        if (img.empty()) {break;}
        float thresh = 0.3;
        std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);
        int classes = 80;
        int width = std::max(1.0f, img.rows * .005f);
        //show detection results
        for (int i=0;i<bbox_vec.size();++i){
            bbox_t b = bbox_vec[i];
            if (b.prob >= conf_thres){
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
        imshow("show",img);
        if (waitKey (5)>= 0){break;}
    }
    return 0;
}