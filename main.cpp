#include "MessageManager.hpp"
#include "SerialPort.hpp"
#include "camera.hpp"
#include "globalParam.hpp"
#include <AimAuto.hpp>
#include "WMIdentify.hpp"
#include "WMPredict.hpp"
#include <UIManager.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <glog/logging.h>
#include <monitor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pthread.h>
#include <ratio>
#include <string>
#include <unistd.h>
#include <filesystem>


#define RECORD_FRAME_COUNT 1800

std::string GetTime()
{
    std::time_t now = std::time(nullptr);
    std::tm *p_tm = std::localtime(&now);
    char time_str[50];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M", p_tm);
    return time_str;
}

// 全局变量参数，这个参数存储着全部的需要的参数
GlobalParam gp;
// 通信类
MessageManager MManager(gp);
#ifndef VIRTUALGRAB
// 相机类
std::string cam_front = "00J59167231"; // 这里的序列号可以根据实际情况进行更改
std::string cam_back = "00E07107258";
Camera camera(gp,cam_front);
Camera camera_back(gp, cam_back);
#endif
Translator auto_temp;
Translator translator;
cv::Mat pic;
cv::Mat pic_back;
bool front,back;
#ifdef NOPORT
const int COLOR = BLUE;
#endif // NOPORT

navInfo_t send_data;
marketCommand_t receive_data;

// 读线程，负责读取串口信息以及取流
void *ReadFunction(void *arg);
// 运算线程，负责对图片进行处理，并且完成后续的要求
void *OperationFunction(void *arg);

int main(int argc, char **argv)
{
    printf("welcome\n");
    // 实例化通信串口类
    SerialPort *serialPort = new SerialPort(argv[1]);
    // 设置通信串口对象初始值
    serialPort->InitSerialPort(int(*argv[2] - '0'), 8, 1, 'N');
#ifndef NOPORT
    MManager.read(auto_temp, *serialPort);
    // 通过电控发来的标志位是0～4还是5～9来确定是红方还是蓝方，其中0～4是红方，5～9是蓝方
    MManager.initParam(auto_temp.message.status / 5 == 0 ? RED : BLUE);
#else
    // 再没有串口的时候直接设定颜色，这句代码可以根据需要进行更改
    MManager.initParam(COLOR);
#endif // NOPORT

    // 输出日志，开始初始化
    pthread_t readThread;
    pthread_t operationThread;

    // 开启线程
    pthread_create(&readThread, NULL, ReadFunction, serialPort);
    pthread_create(&operationThread, NULL, OperationFunction, serialPort);

    // 等待线程结束
    pthread_join(readThread, NULL);
    pthread_join(operationThread, NULL);

    return 0;
}

void *ReadFunction(void *arg) // 读线程
{
#ifdef THREADANALYSIS
    printf("read function init successful\n");
#endif
    // 传入的参数赋给串口，以获得串口数据
    SerialPort *serialPort = (SerialPort *)arg;
    while (1)
    {   
        marketCommand_t temp;
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        MManager.readBoth(auto_temp.message, temp, *serialPort);
        usleep(100);
        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        // printf("read   duration: %ld ms\n", duration);
    }
    return NULL;
}

void *OperationFunction(void *arg)
{   
#ifdef RECORDVIDEO
    cv::VideoWriter *recorder = NULL;
    std::string path = "../video/record/";
    path = path + GetTime() + "/";
    std::filesystem::create_directories(path);
    int coder = cv::VideoWriter::fourcc('H', '2', '6', '4');
    int cnt = RECORD_FRAME_COUNT;
    int idx = 0;
#endif
#ifdef THREADANALYSIS
    printf("operation function init successful\n");
#endif
    SerialPort *serialPort = (SerialPort *)arg;
    // 实例化自瞄类
    AimAuto aim(&gp);
    static WMIdentify WMI(gp);

    static WMPredict WMIPRE(gp);
    // 实例化UI类
    UIManager UI(gp);
    double dt = 0;
    double last_time_stamp = 0;
#ifndef VIRTUALGRAB
    camera.init();
#endif
#ifdef SHOW_FPS
    int fps = 0;
    int frame_count = 0;
    double fps_time_stamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
#ifdef DEBUGMODE
    //=====动态调参使用参数======//
    // 当前按键
    int key = 0;
    // debug时waitKey时间，也就是整体的运行速率
    int debug_t = 1;
    // 储存相机坐标系下的点，用于绘图
    std::deque<cv::Point3f> points3d;
    // 储存当前时间，用于绘图
    std::deque<double> times;

#endif // DEBUGMODE
    //========================//
    int empty_frame_count = 0;
    while (1)
    {
#ifdef RECORDVIDEO
        cv::Mat pic1 = pic.clone();
        if(!pic1.empty()) cv::resize(pic1, pic1, cv::Size(600, 450));
        cnt ++;
        if (cnt > RECORD_FRAME_COUNT && idx <= 50)
        {
            cnt = 0;
            if(recorder != NULL){
                recorder->release();
                delete recorder;
            }
            idx ++;
            recorder = new cv::VideoWriter(path + std::to_string(idx) + ".mp4", coder, 60.0, cv::Size(600, 450), true);
        }
        if(!pic1.empty() && idx <= 50) recorder->write(pic1);
#endif
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
#ifndef NOPORT
        MManager.copy(auto_temp,translator);
#else
        MManager.FakeMessage(translator);
#endif // NOPORT
        if (translator.message.status % 5 != 0)
        {
#ifndef VIRTUALGRAB
            camera.change_attack_mode(ENERGY, gp);
#endif
            gp.attack_mode = ENERGY;
        }
        else
        {
#ifndef VIRTUALGRAB
            camera.change_attack_mode(ARMOR, gp);
#endif
            gp.attack_mode = ARMOR;
        }
        gp.armor_exp_time = translator.message.status / 5 ? gp.red_exp_time : gp.blue_exp_time;

#ifndef NOPORT
        if (translator.message.status / 5 != gp.color)
        {
            gp.initGlobalParam(translator.message.status / 5);
        }
#endif// NOPORT
#ifndef VIRTUALGRAB

        camera.set_param_mult(gp);

        camera.get_pic(&pic);
#else
        MManager.getFrame(pic, translator);
#endif
        // 如果图片为空，不执行
        if (pic.empty()){ 
#ifdef VIRTUALGRAB
            pic = cv::Mat(gp.height, gp.width, CV_8UC3, cv::Scalar(0, 0, 0));
#else       
            empty_frame_count++;
            if (empty_frame_count > 3){
                printf("pic is empty\n");
                exit(0);
            }else{
                pic = cv::Mat(gp.height, gp.width, CV_8UC3, cv::Scalar(0, 0, 0));
            }
#endif
        }else{
            empty_frame_count = 0;
        }
        // 自瞄模式
        if (translator.message.status % 5 == 0)
        {
            double time_stamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            dt = time_stamp - last_time_stamp;
            translator.message.latency = (time_stamp - last_time_stamp) * 1000;
            last_time_stamp = time_stamp;
            aim.auto_aim(pic, translator, dt,front);
            if(front == false && back == true){
                translator.message.allround = true;
            }
            else {
                translator.message.allround = false;
            }
            cout << "front: " << front << " back: " << back << "allround" << translator.message.allround << std::endl;
            MManager.writeBoth(translator.message, send_data,*serialPort);
            // MManager.WriteLogMessage(translator, gp);
// #ifdef DEBUGMODE
#ifdef SHOW_FPS
            cv::putText(pic,"FPS: " + to_string(fps), cv::Point(1000, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
            // printf("FPS: %d  \tLatency: %.3f ms\n", fps, translator.message.latency);
#endif
#ifdef DEBUGMODE
            UI.receive_pic(pic);
            UI.windowsManager(key, debug_t);
            cv::Mat tmp;
            cv::resize(pic, tmp, cv::Size((int)pic.size[1] * gp.resize, (int)pic.size[0] * gp.resize), cv::INTER_LINEAR);
            cv::imshow("aimauto__", tmp);
            // usleep(200 * 1000);
#endif
#ifndef DEBUGMODE
#ifdef SSH
            std::vector<uchar> buf;
            cv::imencode(".jpg", pic, buf); // 将帧编码为 JPEG 格式
            std::string header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
            send(new_socket, header.c_str(), header.size(), 0);
            std::string response = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
            response.insert(response.end(), buf.begin(), buf.end());
            response += "\r\n\r\n";
            send(new_socket, response.c_str(), response.size(), 0);
#endif
#endif
#ifdef DEBUGMODE
            key = cv::waitKey(debug_t);
            if (key == ' ')
                key = cv::waitKey(0);
            if (key == 27 || key == 'q')
                exit(0);
#endif // DEBUGMODE
        } else {
            WMI.identifyWM(pic, translator);
            WMIPRE.StartPredict(translator, gp, WMI);
            MManager.write(translator, *serialPort);
        }
        if (translator.message.status == 99)
            exit(0);
#ifndef NOPORT
#ifdef SSH
        close(new_socket);
        close(server_fd);
#endif
#endif // NOPORT
#ifdef SHOW_FPS
        frame_count++;
        auto now_time_stamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        if (now_time_stamp - fps_time_stamp >= 1)
        {
            fps = frame_count;
            // printf("FPS: %d  \tLatency: %.3f ms\n", frame_count, translator.message.latency);
            frame_count = 0;
            fps_time_stamp = now_time_stamp;
        }
#endif
        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        // printf("option duration: %ld ms\n", duration);
    }
    return NULL;
}
void *DetectFunction(void *arg){
    Detector det(gp);
    camera_back.init();
    camera_back.change_attack_mode(ARMOR, gp);
    camera_back.set(2000);
    int empty_frame_count = 0;
    const int MAX_EMPTY_FRAMES = 10; // 最大空帧数
    while (1) {
        camera_back.get_pic(&pic_back);
        // 检查图片是否为空
        if (pic_back.empty()) {
            empty_frame_count++;
            if (empty_frame_count > MAX_EMPTY_FRAMES) {
                printf("后置相机取流失败，连续空帧数: %d\n", empty_frame_count);
                // 创建黑色图片作为备用
                pic_back = cv::Mat(gp.height, gp.width, CV_8UC3, cv::Scalar(0, 0, 0));
            } else {
                continue; // 跳过这一帧，继续尝试
            }
        } else {
            empty_frame_count = 0; // 重置空帧计数
        }
        det.detect(pic_back, gp.color, back);
        // cv::Mat tmp1;
        // cv::resize(pic_back, tmp1, cv::Size((int)pic_back.size[1] * gp.resize, (int)pic_back.size[0] * gp.resize), cv::INTER_LINEAR);
        // cv::imshow("aimauto__back", tmp1);
        usleep(15000); // 15ms延时
    }
    return NULL;
}