/**
 * @file camera.hpp
 * @author riverflows2333 (liuyuzheng0801@163.com)
 * @brief 相机类声明文件，提供初始化相机、开始取流、设置参数、图像转化并获得图像、改变参数的方法
 * @version 0.1
 * @date 2022-12-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#if !defined(__CAMERA_HPP)
#define __CAMERA_HPP
#include "globalParam.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <glog/logging.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/video.hpp>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class Camera
{
public:
    void *handle;                        //!< 创建句柄，用于指向设备
    int nRet;                            //!< 函数返回码，成功为MV_OK(0x00000000)
    int color;                           //!< 当前颜色，此处为需要击打的敌方装甲板颜色
    int attack_mode;                     //!< 当前击打模式
    unsigned char *pDataForRGB;          //!< 指向取流成功的未经过cvtColor的RGB图片的指针
    MV_CC_DEVICE_INFO_LIST stDeviceList; //!< 设备列表，一般来说只有一个设备
    MV_FRAME_OUT stOutFrame;             //!< 输出的图像，从getImageBuffer中读取到的内容会被存放在这里
    MV_CC_PIXEL_CONVERT_PARAM CvtParam;  //!< 一个参数列表，用于储存ConvertPixelType时使用的全部参数
    MVCC_FLOATVALUE frame_rate;          //!< 获取的帧率信息，用于测试时了解当前帧率
    bool switch_INFO;                    //!< 是否开启所谓INFO级别的log，用于调试
    GlobalParam *gp;

public:
    /**
     * @brief 构造函数，对Camera类初始化、创建句柄并开启设备
     *
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     */
    Camera(GlobalParam &,std::string &);
    /**
     * @brief 相机析构函数，用于在结束进程时销毁句柄，释放内存
     *
     */
    ~Camera();
    /**
     * @brief 相机开始取流
     *
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     * @return int 返回值为MV_OK则成功开启
     */
    int start_camera(GlobalParam &);
    /**
     * @brief 通过类中的参数结构体，配置相机参数，此方法只调用一次
     *
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     * @return int 返回值为MV_OK则为成功
     */
    int set_param_once(GlobalParam &);
    /**
     * @brief 通过类中的参数结构体以及颜色状态，配置相机参数，此方法在整体流程中可用于反复调整调整颜色、击打目标改变时对应需要更改的参数
     *
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     * @return int 返回值为MV_OK则为成功
     */
    int set_param_mult(GlobalParam &);
    /**
     * @brief 获得图片的函数，进行get buffer，然后将获得图片进行convert获得BGR图片，通过scrimg输出
     *
     * @param srcimg 主程序提供的储存获得图片的矩阵
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     * @return int 返回为0则方法正常执行
     */
    int set(int time);
    int get_pic(cv::Mat *srcimg);
    /**
     * @brief 改变颜色这一参数的函数
     *
     * @param input_color 输入新的颜色
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     * @return int 返回值为0则为成功
     */
    int change_color(int, GlobalParam &);
    /**
     * @brief 改变攻击模式这一参数的函数
     *
     * @param input_attack_mode 输入新的攻击模式
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     * @return int 返回值为0则为成功
     */
    int change_attack_mode(int, GlobalParam &);
    /**
     * @brief 请求当前的帧率，避免频繁显示帧率刷屏
     *
     * @param gp 从主函数传入的GlobalParam，从中读取参数，添加原因因架构，详见readme
     * @return int 返回值为0则为成功
     */
    int request_frame_rate(GlobalParam &);
    void init();
    void getFrame(cv::Mat &pic);
};
#endif // __CAMERA_HPP