#include "MessageManager.hpp"
#include "SerialPort.hpp"
#include "camera.hpp"
#include "globalParam.hpp"
#include <AimAuto.hpp>
#include <UIManager.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <glog/logging.h>
#include <monitor.hpp>
#include <pthread.h>
#include <ratio>
#include <string>
#include <unistd.h>
#include <filesystem>

Camera::Camera(GlobalParam &gp,std::string &serial_number)
{
    // 对于Camera中的变量进行初始化
    this->nRet = MV_OK;
    this->handle = NULL;
    this->color = gp.color;
    this->attack_mode = ARMOR;
    this->CvtParam = {0};
    this->stOutFrame = {0};
    this->frame_rate = {0};
    this->switch_INFO = gp.switch_INFO;
    memset(&this->stOutFrame, 0, sizeof(MV_FRAME_OUT));
    memset(&this->stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    this->gp = &gp;
    // 初始化变量统计该方法总的错误数量
    int nRetTotal = 0;
    // 读取设备列表，储存入stDeviceList
    this->nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (stDeviceList.nDeviceNum == 0)
    {
        printf("nDeviceNum == 0\n");
        exit(-1);
    }
// 用序列号查找对应设备
    int found_index = -1;
    for (unsigned int i = 0; i < stDeviceList.nDeviceNum; ++i)
    {
        MV_CC_DEVICE_INFO* pInfo = stDeviceList.pDeviceInfo[i];
        if (pInfo->nTLayerType == MV_USB_DEVICE)
        {
            MV_USB3_DEVICE_INFO* pUsbInfo = &(pInfo->SpecialInfo.stUsb3VInfo);
            std::string sn((char*)pUsbInfo->chSerialNumber);
            if (sn == serial_number)
            {
                found_index = i;
                break;
            }
        }
    }
    if (found_index == -1)
    {
        printf("未找到指定序列号的相机\n");
        exit(-1);
    }

    // 用找到的设备信息创建句柄
    this->nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[found_index]);
    // 依照gp中的cam_index，设置句柄为设备列表中第cam_index个设备(存在第0个)
    // this->nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[gp.cam_index]);
    if (this->nRet != MV_OK)
    {
#ifdef THREADANALYSIS
        printf("CreateHandle ERROR\n");
#endif
        exit(-1);
    }

    // 开启设备
    this->nRet = MV_CC_OpenDevice(handle);
    if (this->nRet != MV_OK)
    {
#ifdef THREADANALYSIS
        printf("OpenDevice ERROR\n");
#endif
        exit(-1);
    }
    printf("cam%d初始化完毕，欢迎使用喵～\n", gp.cam_index);
}

Camera::~Camera()
{
    // 停止取流
    this->nRet = MV_CC_StopGrabbing(this->handle);
    // 关闭设备
    this->nRet = MV_CC_CloseDevice(this->handle);
    // 销毁句柄
    this->nRet = MV_CC_DestroyHandle(this->handle);
    printf("析构中，再见喵～\n");
    // 剩余内容系统自动释放
}

int Camera::start_camera(GlobalParam &gp)
{
    // 开始取流
    this->nRet = MV_CC_StartGrabbing(this->handle);
    if (this->nRet != MV_OK)
    {
#ifdef THREADANALYSIS
        printf("StartGrabbing ERROR\n");
#endif
        exit(-1);
    }
    return 0;
}

int Camera::get_pic(cv::Mat *srcimg)
{
    // chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    this->nRet = MV_CC_GetImageBuffer(handle, &stOutFrame, 400);
    if (this->nRet != MV_OK)
        exit(-1);
    cv::Mat temp;
    temp = cv::Mat(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, CV_8UC1, CvtParam.pSrcData = stOutFrame.pBufAddr);
    if (temp.empty() == 1)
        return -1;
    cv::cvtColor(temp, *srcimg, cv::COLOR_BayerRG2RGB);
    if (NULL != stOutFrame.pBufAddr)
    {
        this->nRet = MV_CC_FreeImageBuffer(handle, &stOutFrame);
        if (this->nRet != MV_OK)
        {
#ifdef THREADANALYSIS
            printf("FreeImageBuffer ERROR\n");
#endif
        }
    }
    // chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    // printf("read   duration: %ld us\n", duration);
    return 0;
}

int Camera::set_param_once(GlobalParam &gp)
{
    this->nRet = MV_CC_SetHeight(this->handle, gp.height);
    this->nRet = MV_CC_SetWidth(this->handle, gp.width);
    // 设置触发模式
#ifdef USETRIGGERMODE
    this->nRet = MV_CC_riggerMode(this->handle, MV_TRIGGER_MODE_ON);
#else
    this->nRet = MV_CC_SetTriggerMode(this->handle, MV_TRIGGER_MODE_OFF);
#endif // USETRIGGERMODE
    this->nRet = MV_CC_SetBalanceWhiteAuto(this->handle, ON);
    this->nRet = MV_CC_SetGamma(this->handle, gp.gamma_value);
    // 设置自动曝光使能
    this->nRet = MV_CC_SetExposureAutoMode(this->handle, gp.enable_auto_exp);
    // 设置曝光时间，默认为蓝方击打装甲板的情况
    this->nRet = MV_CC_SetExposureTime(this->handle, gp.armor_exp_time);
    // 设置自动增益使能
    this->nRet = MV_CC_SetGainMode(this->handle, gp.enable_auto_gain);
    // 设置增益值
    this->nRet = MV_CC_SetGain(this->handle, gp.gain);
    // 设置图像格式为Bayer RG8
    this->nRet = MV_CC_SetPixelFormat(this->handle, gp.pixel_format);
    // 设置帧率
    this->nRet = MV_CC_SetFrameRate(this->handle, gp.frame_rate);
    return 0;
}

int Camera::set_param_mult(GlobalParam &gp)
{
    // this->attack_mode = gp.attack_mode;
    //  初始化变量统计该方法总的错误数量
    //  设置白平衡，依次顺序为BGR
    //  如果当前状态为打击能量机关，输入打能量机关对应的曝光时间
    // printf("exp:%f\n",gp.energy_exp_time);
    if (this->attack_mode == ENERGY)
        this->nRet = MV_CC_SetExposureTime(this->handle, gp.energy_exp_time);
    // 如果当前状态为打击装甲板，输入打装甲板对应的曝光时间
    else if (this->attack_mode == ARMOR)
        this->nRet = MV_CC_SetExposureTime(this->handle, gp.armor_exp_time);
    return 0;
}

int Camera::set(int time){
    this->nRet = MV_CC_SetExposureTime(this->handle,time);
    return 0;
}

int Camera::change_color(int input_color, GlobalParam &gp)
{
    this->color = input_color;
    return 0;
}

int Camera::change_attack_mode(int input_attack_mode, GlobalParam &gp)
{
    this->attack_mode = input_attack_mode;
    return 0;
}

int Camera::request_frame_rate(GlobalParam &gp)
{
    // 读取帧率信息，用于调试
    this->nRet = MV_CC_GetFrameRate(handle, &frame_rate);
    // 打印帧率信息，用于调试
    printf("当前帧率:%f\n", frame_rate.fCurValue);
    return 0;
}

void Camera::init()
{
    // 设置参数
    this->set_param_once(*gp);
    // 打开相机
    this->start_camera(*gp);
}

void Camera::getFrame(cv::Mat &pic)
{
    // 如果不是虚拟取流，先设置相机参数，之后取流
    // #ifdef DEBUGMODE
    this->set_param_mult(*gp);
    // #endif
    this->get_pic(&pic);
    //====去畸变=====//
    // cv::Mat pic_undistort;
    // pic.copyTo(pic_undistort);
    // cv::undistort(pic, pic, gp->K, gp->distort_coeffs);
    // pic_undistort.copyTo(pic);
}