///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
#include "stdafx_ut.h"

#include "RecorderCSV.h"

using namespace Utilities;

// Default constructor initializes the variables
RecorderCSV::RecorderCSV():output_file(){};

// Making sure full stop is used for decimal point separation
struct fullstop : std::numpunct<char> {
	char do_decimal_point() const { return '.'; }
};

// Opening the file and preparing the header for it
bool RecorderCSV::Open(std::string output_file_name, bool is_sequence, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_face_landmarks, int num_model_modes, int num_eye_landmarks, const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg)
{

	output_file.open(output_file_name, std::ios_base::out);
	output_file.imbue(std::locale(output_file.getloc(), new fullstop));

	if (!output_file.is_open())
		return false;

	this->is_sequence = is_sequence;

	// Set up what we are recording
	this->output_2D_landmarks = output_2D_landmarks;
	this->output_3D_landmarks = output_3D_landmarks;
	this->output_AUs = output_AUs;
	this->output_gaze = output_gaze;
	this->output_model_params = output_model_params;
	this->output_pose = output_pose;

	this->au_names_class = au_names_class;
	this->au_names_reg = au_names_reg;

	// Different headers if we are writing out the results on a sequence or an individual image
	if(this->is_sequence)
	{
		output_file << "frame,face_id,timestamp,confidence,success";
	}
	else
	{
		output_file << "face,confidence";
	}

	if (output_gaze)
	{
		output_file << ",gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z,gaze_angle_x,gaze_angle_y";

		for (int i = 0; i < num_eye_landmarks; ++i)
		{
			output_file << ",eye_lmk_x_" << i;
		}
		for (int i = 0; i < num_eye_landmarks; ++i)
		{
			output_file << ",eye_lmk_y_" << i;
		}

		for (int i = 0; i < num_eye_landmarks; ++i)
		{
			output_file << ",eye_lmk_X_" << i;
		}
		for (int i = 0; i < num_eye_landmarks; ++i)
		{
			output_file << ",eye_lmk_Y_" << i;
		}
		for (int i = 0; i < num_eye_landmarks; ++i)
		{
			output_file << ",eye_lmk_Z_" << i;
		}
	}

	if (output_pose)
	{
		output_file << ",pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz";
	}

	if (output_2D_landmarks)
	{
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ",x_" << i;
		}
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ",y_" << i;
		}
	}

	if (output_3D_landmarks)
	{
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ",X_" << i;
		}
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ",Y_" << i;
		}
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ",Z_" << i;
		}
	}

	// Outputting model parameters (rigid and non-rigid), the first parameters are the 6 rigid shape parameters, they are followed by the non rigid shape parameters
	if (output_model_params)
	{
		output_file << ",p_scale,p_rx,p_ry,p_rz,p_tx,p_ty";
		for (int i = 0; i < num_model_modes; ++i)
		{
			output_file << ",p_" << i;
		}
	}

	if (output_AUs)
	{
		std::sort(this->au_names_reg.begin(), this->au_names_reg.end());
		for (std::string reg_name : this->au_names_reg)
		{
			output_file << "," << reg_name << "_r";
		}

		std::sort(this->au_names_class.begin(), this->au_names_class.end());
		for (std::string class_name : this->au_names_class)
		{
			output_file << "," << class_name << "_c";
		}
	}

	output_file << "\n";

	return true;

}

std::string RecorderCSV::WriteLine(int face_id, int frame_num, double time_stamp, bool landmark_detection_success, double landmark_confidence,
	const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D, const cv::Mat_<float>& pdm_model_params, const cv::Vec6f& rigid_shape_params, cv::Vec6f& pose_estimate,
	const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d,
	const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences, bool output_to_csv_file)
{
	if (!output_file.is_open())
	{
		std::cout << "The output CSV file is not open, exiting" << std::endl;
		exit(1);
	}

	std::ostringstream csv_line = std::ostringstream();

	// Making sure fixed and not scientific notation is used
	csv_line << std::fixed;
	csv_line << std::noshowpoint;
	if(is_sequence)
	{
		
		csv_line << std::setprecision(3);
		csv_line << frame_num << "," << face_id << "," << time_stamp;
		csv_line << std::setprecision(2);
		csv_line << "," << landmark_confidence;
		csv_line << std::setprecision(0);
		csv_line << "," << landmark_detection_success;
	}
	else
	{
		csv_line << std::setprecision(3);
		csv_line << face_id << "," << landmark_confidence;
	}
	// Output the estimated gaze
	if (output_gaze)
	{
		csv_line << std::setprecision(6);
		csv_line << "," << gazeDirection0.x << "," << gazeDirection0.y << "," << gazeDirection0.z
			<< "," << gazeDirection1.x << "," << gazeDirection1.y << "," << gazeDirection1.z;

		// Output gaze angle (same format as head pose angle)
		csv_line << std::setprecision(3);
		csv_line << "," << gaze_angle[0] << "," << gaze_angle[1];

		// Output the 2D eye landmarks
		csv_line << std::setprecision(1);
		for (auto eye_lmk : eye_landmarks2d)
		{
			csv_line << "," << eye_lmk.x;
		}

		for (auto eye_lmk : eye_landmarks2d)
		{
			csv_line << "," << eye_lmk.y;
		}

		// Output the 3D eye landmarks
		for (auto eye_lmk : eye_landmarks3d)
		{
			csv_line << "," << eye_lmk.x;
		}

		for (auto eye_lmk : eye_landmarks3d)
		{
			csv_line << "," << eye_lmk.y;
		}

		for (auto eye_lmk : eye_landmarks3d)
		{
			csv_line << "," << eye_lmk.z;
		}
	}

	// Output the estimated head pose
	if (output_pose)
	{
		csv_line << std::setprecision(1);
		csv_line << "," << pose_estimate[0] << "," << pose_estimate[1] << "," << pose_estimate[2];
		csv_line << std::setprecision(3);
		csv_line << "," << pose_estimate[3] << "," << pose_estimate[4] << "," << pose_estimate[5];
	}

	// Output the detected 2D facial landmarks
	if (output_2D_landmarks)
	{
		csv_line.precision(1);
		// Output the 2D eye landmarks
		for (auto lmk : landmarks_2D)
		{
			csv_line << "," << lmk;
		}
	}

	// Output the detected 3D facial landmarks
	if (output_3D_landmarks)
	{
		csv_line.precision(1);
		// Output the 2D eye landmarks
		for (auto lmk : landmarks_3D)
		{
			csv_line << "," << lmk;
		}
	}

	if (output_model_params)
	{
		csv_line.precision(3);
		for (int i = 0; i < 6; ++i)
		{
			csv_line << "," << rigid_shape_params[i];
		}
		// Output the non_rigid shape parameters
		for (auto lmk : pdm_model_params)
		{
			csv_line << "," << lmk;
		}
	}

	if (output_AUs)
	{

		// write out ar the correct index
		csv_line.precision(2);
		for (std::string au_name : au_names_reg)
		{
			for (auto au_reg : au_intensities)
			{
				if (au_name.compare(au_reg.first) == 0)
				{
					csv_line << "," << au_reg.second;
					break;
				}
			}
		}

		if (au_intensities.size() == 0)
		{
			for (size_t p = 0; p < au_names_reg.size(); ++p)
			{
				csv_line << ",0";
			}
		}

		csv_line.precision(1);
		// write out ar the correct index
		for (std::string au_name : au_names_class)
		{
			for (auto au_class : au_occurences)
			{
				if (au_name.compare(au_class.first) == 0)
				{
					csv_line << "," << au_class.second;
					break;
				}
			}
		}

		if (au_occurences.size() == 0)
		{
			for (size_t p = 0; p < au_names_class.size(); ++p)
			{
				csv_line << ",0";
			}
		}
	}
	csv_line << "\n";

	if (output_to_csv_file) {
		output_file << csv_line.str();
	}

	return csv_line.str();
}

// Closing the file and cleaning up
void RecorderCSV::Close()
{
	output_file.close();
}
