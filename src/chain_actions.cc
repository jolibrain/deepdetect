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

namespace dd
{

  int ImgsCropAction::apply(APIData &model_out,
			    std::unordered_map<std::string,APIData> &actions_data)
    {
      std::cerr << "[chain] applying crops action\n";
      std::vector<APIData> vad = model_out.getv("predictions");
      //std::cerr << "ad_in has inputs=" << model_out.has("input") << std::endl;
      std::vector<cv::Mat> imgs = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
      std::vector<std::pair<int,int>> imgs_size = model_out.getobj("input").get("imgs_size").get<std::vector<std::pair<int,int>>>();
      /*std::cerr << "imgs_size size=" << imgs_size.size() << std::endl;
	std::cerr << "vad size=" << vad.size() << " / imgs size=" << imgs.size() << std::endl;*/
      std::vector<std::string> cropped_imgs;
      std::vector<std::string> bbox_ids;

      double bratio = 0.015; //TODO: action parameter

      std::vector<APIData> cvad; // TODO: fill out with added ids
      
      //TODO: get image batch
      for (size_t i=0;i<vad.size();i++)
	{
	  std::string uri = vad.at(i).get("uri").get<std::string>();
	  
	  //TODO: crop from cv::Mat
	  cv::Mat img = imgs.at(i);
	  int orig_cols = imgs_size.at(i).second;
	  int orig_rows = imgs_size.at(i).first;
	  
	  std::vector<APIData> ad_cls = vad.at(i).getv("classes");
	  std::vector<APIData> cad_cls;
	  //std::cerr << "classes size=" << ad_cls.size() << std::endl;

	  for (size_t j=0;j<ad_cls.size();j++)
	    {	      
	      //std::cerr << "cls has bbox=" << ad_cls.at(j).has("bbox") << std::endl;
	      APIData bbox = ad_cls.at(j).getobj("bbox");
	      //std::cerr << "bbox size=" << bbox.size() << std::endl;

	      // adding modified object chain id
	      std::string bbox_id = genid(uri,"bbox"+std::to_string(j));
	      bbox_ids.push_back(bbox_id);
	      APIData ad_cid;
	      //ad_cid.add("chainid",bbox_id);
	      ad_cls.at(j).add(bbox_id,ad_cid);
	      cad_cls.push_back(ad_cls.at(j));
	      
	      double xmin = bbox.get("xmin").get<double>() / orig_cols * img.cols;
	      double ymin = bbox.get("ymin").get<double>() / orig_rows * img.rows;
	      double xmax = bbox.get("xmax").get<double>() / orig_cols * img.cols;
	      double ymax = bbox.get("ymax").get<double>() / orig_rows * img.rows;

	      /*std::cerr << "xmin=" << xmin << " / xmax=" << xmax << " / ymin= " << ymin << " / ymax=" << ymax << std::endl;
		std::cerr << "img cols=" << img.cols << " / img rows=" << img.rows << std::endl;*/
	      
	      double deltax = bratio * (xmax-xmin);
	      double deltay = bratio * (ymin-ymax);

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
	      //cv::imwrite("crop" + std::to_string(j) + ".png",cropped_img);
	      //debug
	      
	      //TODO: serialize crop into string (will be auto read by read_element in imginputconn)
	      std::vector<unsigned char> cropped_img_ser;
	      bool str_encoding = cv::imencode(".png",cropped_img,cropped_img_ser);
	      if (!str_encoding)
		{
		  //TODO: exception
		  std::cerr << "[action] crop encoding error\n";
		}
	      std::string cropped_img_str = std::string(cropped_img_ser.begin(),cropped_img_ser.end());
	      cropped_imgs.push_back(cropped_img_str);
	    }
	  APIData ccls;
	  ccls.add("classes",cad_cls);
	  cvad.push_back(ccls);
	}
      //TODO: store serialized crops into actions_out
      APIData action_out;
      action_out.add("data",cropped_imgs);
      action_out.add("cids",bbox_ids);
      actions_data.insert(std::pair<std::string,APIData>(_action_type,action_out));

      // updated model data with chain ids
      model_out.add("predictions",cvad);

      return 0;
    }


  int ClassFilter::apply(APIData &model_out,
			 std::unordered_map<std::string,APIData> &actions_data)
  {
    std::cerr << "[chain] applying filter action\n";

    if (!_params.has("classes"))
      {
	std::cerr << "Action filter is missing classes parameter\n";
	return 1;
	//TODO: throw
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
    
    return 0;
  }

} // end of namespace
