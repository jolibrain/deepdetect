/**
 * DeepDetect
 * Copyright (c) 2019 Emmanuel Benazera
 * Author: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "chain_actions.h"
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include "utils/utils.hpp"

namespace dd
{

  void ImgsCropAction::apply(APIData &model_out,
			     std::vector<APIData> &actions_data)
    {
      std::cerr << "[chain] applying crops action\n";
      std::vector<APIData> vad = model_out.getv("predictions");
      std::vector<cv::Mat> imgs = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
      std::vector<std::pair<int,int>> imgs_size = model_out.getobj("input").get("imgs_size").get<std::vector<std::pair<int,int>>>();
      std::vector<std::string> cropped_imgs;
      std::vector<std::string> bbox_ids;

      //double bratio = 0.015; //TODO: action parameter

      std::vector<APIData> cvad;
      
      // iterate image batch
      for (size_t i=0;i<vad.size();i++)
	{
	  std::string uri = vad.at(i).get("uri").get<std::string>();
	  std::cerr << "crop on URI=" << uri << std::endl;
	  
	  cv::Mat img = imgs.at(i);
	  int orig_cols = imgs_size.at(i).second;
	  int orig_rows = imgs_size.at(i).first;
	  
	  std::vector<APIData> ad_cls = vad.at(i).getv("classes");
	  std::vector<APIData> cad_cls;
	  
	  // iterate bboxes per image
	  for (size_t j=0;j<ad_cls.size();j++)
	    {	      
	      APIData bbox = ad_cls.at(j).getobj("bbox");
	      if (bbox.empty())
		throw ActionBadParamException("crop action cannot find bbox object for uri " + uri);
	      
	      // adding modified object chain id
	      std::string bbox_id = genid(uri,"bbox"+std::to_string(j));
	      std::cerr << "bbox_id=" << bbox_id << std::endl;
	      bbox_ids.push_back(bbox_id);
	      APIData ad_cid;
	      ad_cls.at(j).add(bbox_id,ad_cid);
	      cad_cls.push_back(ad_cls.at(j));
	      
	      double xmin = bbox.get("xmin").get<double>() / orig_cols * img.cols;
	      double ymin = bbox.get("ymin").get<double>() / orig_rows * img.rows;
	      double xmax = bbox.get("xmax").get<double>() / orig_cols * img.cols;
	      double ymax = bbox.get("ymax").get<double>() / orig_rows * img.rows;

	      /*std::cerr << "xmin=" << xmin << " / xmax=" << xmax << " / ymin= " << ymin << " / ymax=" << ymax << std::endl;
		std::cerr << "img cols=" << img.cols << " / img rows=" << img.rows << std::endl;*/
	      
	      /*double deltax = bratio * (xmax-xmin);
		double deltay = bratio * (ymin-ymax);*/

	      //std::cerr << "deltax=" << deltax << " / deltay=" << deltay << " / xmax-xmin=" << xmax-xmin << " / ymin-ymax=" << ymin-ymax << std::endl;
	      
	      /*double cxmin = std::max(0.0,xmin-deltax);
	      double cxmax = std::min(static_cast<double>(img.cols),xmax+deltax);
	      double cymax = std::max(0.0,ymin-deltay);
	      double cymin = std::min(static_cast<double>(img.rows),ymax+deltay);

	      std::cerr << "cxmin=" << cxmin << " / cxmax=" << cxmax << " / ymin= " << cymin << " / ymax=" << cymax << std::endl;*/
	      
	      //cv::Rect roi(cxmin,cymax,cxmax-cxmin,cymax-cymin);
	      cv::Rect roi(xmin,ymax,xmax-xmin,ymin-ymax);
	      //std::cerr << "roi x=" << roi.x << " / y=" << roi.y << " / width=" << roi.width << " / height=" << roi.height << std::endl;
	      cv::Mat cropped_img = img(roi);

	      //debug
	      /*std::string puri = dd_utils::split(uri,'/').back();
	      std::cerr << "writing crop=" << "crop_" + puri + "_" + std::to_string(j) + ".png\n";
	      cv::imwrite("crop_" + puri + "_" + std::to_string(j) + ".png",cropped_img);*/
	      //debug
	      
	      // serialize crop into string (will be auto read by read_element in imginputconn)
	      std::vector<unsigned char> cropped_img_ser;
	      bool str_encoding = cv::imencode(".png",cropped_img,cropped_img_ser);
	      if (!str_encoding)
		throw ActionInternalException("crop encoding error for uri " + uri);
	      std::string cropped_img_str = std::string(cropped_img_ser.begin(),cropped_img_ser.end());
	      cropped_imgs.push_back(cropped_img_str);
	    }
	  APIData ccls;
	  ccls.add("uri",uri);
	  ccls.add("classes",cad_cls);
	  cvad.push_back(ccls);
	}
      // store serialized crops into action output store
      APIData action_out;
      action_out.add("data",cropped_imgs);
      action_out.add("cids",bbox_ids);
      //actions_data.insert(std::pair<std::string,APIData>(_action_type,action_out));
      actions_data.push_back(action_out);
      
      // updated model data with chain ids
      model_out.add("predictions",cvad);
    }

  void ClassFilter::apply(APIData &model_out,
			  std::vector<APIData> &actions_data)
  {
    std::cerr << "[chain] applying filter action\n";
    if (!_params.has("classes"))
      {
	throw ActionBadParamException("filter action is missing classes parameter");
      }
    std::vector<std::string> on_classes = _params.get("classes").get<std::vector<std::string>>();
    std::unordered_set<std::string> on_classes_us;
    for (auto s: on_classes)
      on_classes_us.insert(s);
    std::unordered_set<std::string>::const_iterator usit;
    
    std::vector<APIData> vad = model_out.getv("predictions");
    std::vector<APIData> cvad;
    
    for (size_t i=0;i<vad.size();i++)
      {
	std::vector<APIData> ad_cls = vad.at(i).getv("classes");
	std::vector<APIData> cad_cls;

	for (size_t j=0;j<ad_cls.size();j++)
	  {
	    std::string cat = ad_cls.at(j).get("cat").get<std::string>();
	    std::cerr << "cat=" << cat << std::endl;
	    if ((usit=on_classes_us.find(cat))!=on_classes_us.end())
	      {
		std::cerr << "filtered cat=" << cat << std::endl;
		cad_cls.push_back(ad_cls.at(j));
	      }
	  }
	APIData ccls;
	ccls.add("classes",cad_cls);
	cvad.push_back(ccls);
      }
    
    // updated model data
    model_out.add("predictions",cvad);
  }

} // end of namespace
