/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>

#include <condition_variable>

#include <opencv2/core/core.hpp>

#include <librealsense2/rs.hpp>
#include "librealsense2/rsutil.h"

#include <System.h>

#include <fstream>

using namespace std;

bool b_continue_session;

void exit_loop_handler(int s){
    std::cout << "Finishing session" << endl;
    b_continue_session = false;

}

std::vector<double> v_t_track;
std::vector<double> v_t_fps;
double compute_mean(const std::vector<double> &v);
double compute_median(std::vector<double> v);
double compute_total(const std::vector<double> &v);
void save_images(const cv::Mat &rgb_image, const cv::Mat &depth_image, double &timestamp, const std::string &output_dir);

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);


void interpolateData(const std::vector<double> &vBase_times,
                     std::vector<rs2_vector> &vInterp_data, std::vector<double> &vInterp_times,
                     const rs2_vector &prev_data, const double &prev_time);

rs2_vector interpolateMeasure(const double target_time,
                              const rs2_vector current_data, const double current_time,
                              const rs2_vector prev_data, const double prev_time);

static rs2_option get_sensor_option(const rs2::sensor& sensor)
{
    // Sensors usually have several options to control their properties
    //  such as Exposure, Brightness etc.

    std::cout << "Sensor supports the following options:\n" << std::endl;

    // The following loop shows how to iterate over all available options
    // Starting from 0 until RS2_OPTION_COUNT (exclusive)
    for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
    {
        rs2_option option_type = static_cast<rs2_option>(i);
        //SDK enum types can be streamed to get a string that represents them
        std::cout << "  " << i << ": " << option_type;

        // To control an option, use the following api:

        // First, verify that the sensor actually supports this option
        if (sensor.supports(option_type))
        {
            std::cout << std::endl;

            // Get a human readable description of the option
            const char* description = sensor.get_option_description(option_type);
            std::cout << "       Description   : " << description << std::endl;

            // Get the current value of the option
            float current_value = sensor.get_option(option_type);
            std::cout << "       Current Value : " << current_value << std::endl;

            //To change the value of an option, please follow the change_sensor_option() function
        }
        else
        {
            std::cout << " is not supported" << std::endl;
        }
    }

    uint32_t selected_sensor_option = 0;
    return static_cast<rs2_option>(selected_sensor_option);
}

int main(int argc, char **argv) {

    if (argc < 4 || argc > 5) {
        cerr << endl
             << "Usage: ./rgbd_realsense_D455i path_to_vocabulary path_to_settings path_to_output_dir (trajectory_file_name)"
             << endl;
        return 1;
    }

    string output_dir = string(argv[3]);
    if (output_dir.back() != '/') {
        output_dir += "/";
    }

    string file_name;
    bool bFileName = false;

    if (argc == 5) {
        file_name = string(argv[argc - 1]);
        bFileName = true;
    }

    string rgb_out_dir = output_dir + "rgb/";
    string depth_out_dir = output_dir + "depth/";
    
    string mkdir_cmd = "mkdir -p " + rgb_out_dir + " " + depth_out_dir;
    int dir_status = system(mkdir_cmd.c_str());

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    b_continue_session = true;

    double offset = 0; // ms

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        return 0;
    }
    else
        selected_device = devices[0];

    std::vector<rs2::sensor> sensors = selected_device.query_sensors();
    int index = 0;
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors)
        if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
            ++index;
            if (index == 1) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
                sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_LIMIT,50000);
                sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1); // emitter on for depth information
            }
            // std::cout << "  " << index << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
            get_sensor_option(sensor);
            if (index == 2){
                // RGB camera
                sensor.set_option(RS2_OPTION_EXPOSURE,80.f);

            }

            if (index == 3){
                sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION,0);
            }

        }

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    // RGB stream
    cfg.enable_stream(RS2_STREAM_COLOR,640, 480, RS2_FORMAT_RGB8, 30);

    // Depth stream
    // cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH,640, 480, RS2_FORMAT_Z16, 30);

    // IMU stream
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F); //, 250); // 63
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F); //, 400);

    // IMU callback
    std::mutex imu_mutex;
    std::condition_variable cond_image_rec;

    vector<double> v_accel_timestamp;
    vector<rs2_vector> v_accel_data;
    vector<double> v_gyro_timestamp;
    vector<rs2_vector> v_gyro_data;

    double prev_accel_timestamp = 0;
    rs2_vector prev_accel_data;
    double current_accel_timestamp = 0;
    rs2_vector current_accel_data;
    vector<double> v_accel_timestamp_sync;
    vector<rs2_vector> v_accel_data_sync;

    cv::Mat imCV, depthCV;
    int width_img, height_img;
    double timestamp_image = -1.0;
    bool image_ready = false;
    int count_im_buffer = 0; // count dropped frames

    // start and stop just to get necessary profile
    rs2::pipeline_profile pipe_profile = pipe.start(cfg);
    pipe.stop();

    // Align depth and RGB frames
    //Pipeline could choose a device that does not have a color stream
    //If there is no color stream, choose to align depth to another stream
    rs2_stream align_to = find_stream_to_align(pipe_profile.get_streams());

    // Create a rs2::align object.
    // rs2::align allows us to perform alignment of depth frames to others frames
    //The "align_to" is the stream type to which we plan to align depth frames.
    rs2::align align(align_to);
    rs2::frameset fsSLAM;

    auto imu_callback = [&](const rs2::frame& frame)
    {
        std::unique_lock<std::mutex> lock(imu_mutex);

        if(rs2::frameset fs = frame.as<rs2::frameset>())
        {
            count_im_buffer++;

            double new_timestamp_image = fs.get_timestamp()*1e-3;
            if(abs(timestamp_image-new_timestamp_image)<0.001){
                count_im_buffer--;
                return;
            }

            if (profile_changed(pipe.get_active_profile().get_streams(), pipe_profile.get_streams()))
            {
                //If the profile was changed, update the align object, and also get the new device's depth scale
                pipe_profile = pipe.get_active_profile();
                align_to = find_stream_to_align(pipe_profile.get_streams());
                align = rs2::align(align_to);
            }

            //Align depth and rgb takes long time, move it out of the interruption to avoid losing IMU measurements
            fsSLAM = fs;

            /*
            //Get processed aligned frame
            auto processed = align.process(fs);


            // Trying to get both other and aligned depth frames
            rs2::video_frame color_frame = processed.first(align_to);
            rs2::depth_frame depth_frame = processed.get_depth_frame();
            //If one of them is unavailable, continue iteration
            if (!depth_frame || !color_frame) {
                std::cout << "Not synchronized depth and image\n";
                return;
            }


            imCV = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
            depthCV = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

            cv::Mat depthCV_8U;
            depthCV.convertTo(depthCV_8U,CV_8U,0.01);
            cv::imshow("depth image", depthCV_8U);*/

            timestamp_image = fs.get_timestamp()*1e-3;
            image_ready = true;

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {
                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                v_accel_data_sync.push_back(current_accel_data);
                v_accel_timestamp_sync.push_back(target_time);
            }

            lock.unlock();
            cond_image_rec.notify_all();
        }
    };


    pipe_profile = pipe.start(cfg, imu_callback);


    rs2::stream_profile cam_stream = pipe_profile.get_stream(RS2_STREAM_COLOR);


    rs2_intrinsics intrinsics_cam = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();
    width_img = intrinsics_cam.width;
    height_img = intrinsics_cam.height;
    std::cout << " fx = " << intrinsics_cam.fx << std::endl;
    std::cout << " fy = " << intrinsics_cam.fy << std::endl;
    std::cout << " cx = " << intrinsics_cam.ppx << std::endl;
    std::cout << " cy = " << intrinsics_cam.ppy << std::endl;
    std::cout << " height = " << intrinsics_cam.height << std::endl;
    std::cout << " width = " << intrinsics_cam.width << std::endl;
    std::cout << " Coeff = " << intrinsics_cam.coeffs[0] << ", " << intrinsics_cam.coeffs[1] << ", " <<
    intrinsics_cam.coeffs[2] << ", " << intrinsics_cam.coeffs[3] << ", " << intrinsics_cam.coeffs[4] << ", " << std::endl;
    std::cout << " Model = " << intrinsics_cam.model << std::endl;

#ifdef COMPILEDWITHC14
    std::chrono::steady_clock::time_point start_init = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point start_init = std::chrono::monotonic_clock::now();
#endif
 
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD, true, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    b_continue_session = true;

    double timestamp;
    cv::Mat im, depth;

    double t_resize = 0.f;
    double t_track = 0.f;
    double t_init = 0.f;
    rs2::frameset fs;

    int frame_id = 0;
    double min_fps = std::numeric_limits<double>::max();
    double max_fps = 0;

#ifdef COMPILEDWITHC14
    std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point end_init = std::chrono::monotonic_clock::now();
#endif
    t_init = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end_init - start_init).count();


    while (!SLAM.isShutDown() && b_continue_session)
    {
        {
            std::unique_lock<std::mutex> lk(imu_mutex);
            if(!image_ready)
                cond_image_rec.wait(lk);

#ifdef COMPILEDWITHC14
            std::chrono::steady_clock::time_point time_Start_Process = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point time_Start_Process = std::chrono::monotonic_clock::now();
#endif

            fs = fsSLAM;

            if(count_im_buffer>1)
                std::cout << count_im_buffer -1 << " dropped frs\n";
            count_im_buffer = 0;

            timestamp = timestamp_image;
            im = imCV.clone();
            depth = depthCV.clone();

            image_ready = false;
        }

        // Perform alignment here
        auto processed = align.process(fs);

        // Trying to get both other and aligned depth frames
        rs2::video_frame color_frame = processed.first(align_to);
        rs2::depth_frame depth_frame = processed.get_depth_frame();

        im = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        depth = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

        /*cv::Mat depthCV_8U;
        depthCV.convertTo(depthCV_8U,CV_8U,0.01);
        cv::imshow("depth image", depthCV_8U);*/

        double timestamp = fs.get_timestamp() * 1e-3; 
        save_images(im, depth, timestamp, output_dir);


        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC14
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            cv::resize(depth, depth, cv::Size(width, height));

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC14
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
            std::cout << "[Resize Time] " << t_resize << " ms" << std::endl;
#endif
        }

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t_Start_Track = std::chrono::steady_clock::now();
    #else
        std::chrono::monotonic_clock::time_point t_Start_Track = std::chrono::monotonic_clock::now();
    #endif
#endif
        // Pass the image to the SLAM system
        SLAM.TrackRGBD(im, depth, timestamp); //, vImuMeas); depthCV

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t_End_Track = std::chrono::steady_clock::now();
    #else
        std::chrono::monotonic_clock::time_point t_End_Track = std::chrono::monotonic_clock::now();
    #endif
    t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Track - t_Start_Track).count();
    SLAM.InsertTrackTime(t_track);
#endif

        double fps = 1000.0/t_track;        
        if (fps > max_fps) max_fps = fps;
        if (fps < min_fps) min_fps = fps;

        v_t_track.push_back(t_track);
        v_t_fps.push_back(fps);

        std::cout<< "Frame " << frame_id << " Tracking time: " << t_track << "ms, FPS: " << fps << std::endl;
        frame_id++;
    }
    
    std::cout << "Saving keyframe trajectory..." << std::endl;
    SLAM.SaveTrajectoryTUM(output_dir + "CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM(output_dir + "KeyFrameTrajectory.txt");
    
    std::cout << "Time initialized: " << t_init << " ms" << std::endl;
    std::cout << "Total tracking time: " << compute_total(v_t_track) << " ms" << std::endl;
    std::cout << "Mean tracking time: " << compute_mean(v_t_track) << " ms" << std::endl;
    std::cout << "Median tracking time: " << compute_median(v_t_track) << " ms" << std::endl;
    std::cout << "Mean FPS: " << compute_mean(v_t_fps) << std::endl;
    std::cout << "Max FPS: " << max_fps << std::endl;
    std::cout << "Min FPS: " << min_fps << std::endl;

    std::string time_stats_path = output_dir + "tracking_stats.txt";
    std::ofstream out_file(time_stats_path);

    if (out_file.is_open())
    {
        out_file << "Time Initialized: " << t_init << " ms" << std::endl;
        out_file << "Total Tracking: " << compute_total(v_t_track) << " ms" << std::endl;
        out_file << "Median tracking time: " << compute_median(v_t_track) << " ms" << std::endl;
        out_file << "Mean tracking time: " << compute_mean(v_t_track) << " ms" << std::endl;
        out_file << "Mean FPS: " << compute_mean(v_t_fps) << std::endl;
        out_file << "Max FPS: " << max_fps << std::endl;
        out_file << "Min FPS: " << min_fps << std::endl;
        out_file.close();
        std::cout << "Tracking time statistics saved to: " << time_stats_path << std::endl;
    }
    
    std::cout << "System shutdown!\n";
    SLAM.Shutdown();
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}


bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}

double compute_mean(const std::vector<double> &v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

double compute_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();

    return (n % 2 == 0) ? ((v[n / 2 - 1] + v[n / 2]) / 2.0) : v[n / 2];
}

double compute_total(const std::vector<double> &v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}

// Save RGB and Depth images with timestamp
void save_images(const cv::Mat &rgb_image, const cv::Mat &depth_image, double &timestamp, const std::string &output_dir){
    
    std::ostringstream oss_timestamp;
    oss_timestamp << std::fixed << std::setprecision(6) << timestamp;
    std::string ts_str = oss_timestamp.str();

    std::string rgb_file = output_dir + "rgb/" + ts_str + ".png";
    std::string depth_file = output_dir + "depth/" + ts_str + ".png";

    cv::Mat rgb_bgr;
    cv::cvtColor(rgb_image, rgb_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(rgb_file, rgb_bgr);

    cv::Mat depth_image_8u;
    depth_image.convertTo(depth_image_8u, CV_8U, 0.01);
    cv::imwrite(depth_file, depth_image_8u);
    
    std::cout << "Saved images: " << rgb_file << ", " << depth_file << std::endl;
}