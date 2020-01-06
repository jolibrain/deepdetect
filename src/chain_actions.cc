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

#ifdef USE_DLIB
#include "backends/dlib/dlib_actions.h"
#endif

namespace dd
{
  void ImgsCropAction::apply(APIData &model_out,
			     ChainData &cdata)
    {
      std::vector<APIData> vad = model_out.getv("predictions");
      std::vector<cv::Mat> imgs = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
      std::vector<std::pair<int,int>> imgs_size = model_out.getobj("input").get("imgs_size").get<std::vector<std::pair<int,int>>>();
      std::vector<cv::Mat> cropped_imgs;
      std::vector<std::string> bbox_ids;

      // check for action parameters
      double bratio = 0.0;
      if (_params.has("padding_ratio"))
	bratio = _params.get("padding_ratio").get<double>(); // e.g. 0.055
      std::vector<APIData> cvad;

      bool save_crops = false;
      if (_params.has("save_crops"))
	save_crops = _params.get("save_crops").get<bool>();

      std::string save_path;
      if (_params.has("save_path"))
	save_path = _params.get("save_path").get<std::string>() + "/";
      
      // iterate image batch
      for (size_t i=0;i<vad.size();i++)
	{
	  std::string uri = vad.at(i).get("uri").get<std::string>();
	  
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

	      // adding bbox id
	      std::string bbox_id = genid(uri,"bbox"+std::to_string(j));
	      bbox_ids.push_back(bbox_id);
	      APIData ad_cid;
	      ad_cls.at(j).add(bbox_id,ad_cid);
	      cad_cls.push_back(ad_cls.at(j));

	      double xmin = bbox.get("xmin").get<double>() / orig_cols * img.cols;
	      double ymin = bbox.get("ymin").get<double>() / orig_rows * img.rows;
	      double xmax = bbox.get("xmax").get<double>() / orig_cols * img.cols;
	      double ymax = bbox.get("ymax").get<double>() / orig_rows * img.rows;
	      
	      double deltax = bratio * (xmax - xmin);
	      double deltay = bratio * (ymax - ymin);

	      double cxmin = std::max(0.0,xmin-deltax);
	      double cxmax = std::min(static_cast<double>(img.cols),xmax+deltax);
	      double cymax = std::max(0.0,ymax-deltay);
	      double cymin = std::min(static_cast<double>(img.rows),ymin+deltay);

	      cv::Rect roi(cxmin,cymax,cxmax-cxmin,cymin-cymax);
	      cv::Mat cropped_img = img(roi);

	      // save crops if requested
	      if (save_crops)
		{
		  std::string puri = dd_utils::split(uri,'/').back();
		  cv::imwrite(save_path + "crop_" + puri + "_" + std::to_string(j) + ".png",cropped_img);
		}
	      cropped_imgs.push_back(std::move(cropped_img));
	    }
	  APIData ccls;
	  ccls.add("uri",uri);
	  if (vad.at(i).has("index_uri"))
	    ccls.add("index_uri",vad.at(i).get("index_uri").get<std::string>());
	  ccls.add("classes",cad_cls);
	  cvad.push_back(ccls);
	}
      // store crops into action output store
      APIData action_out;
      action_out.add("data_raw_img",cropped_imgs);
      action_out.add("cids",bbox_ids);
      cdata.add_action_data(_action_id,action_out);      
      
      // updated model data with chain ids
      model_out.add("predictions",cvad);
    }

  void ImgsRotateAction::apply(APIData &model_out,
			       ChainData &cdata)
  {
    // get label
    std::vector<APIData> vad = model_out.getv("predictions");
    std::vector<cv::Mat> imgs = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
    std::vector<std::pair<int,int>> imgs_size = model_out.getobj("input").get("imgs_size").get<std::vector<std::pair<int,int>>>();
    std::vector<cv::Mat> rimgs;
    std::vector<std::string> uris;

    // check for action parameters
    std::string orientation = "relative"; // other: absolute
    if (_params.has("orientation"))
      orientation = _params.get("orientation").get<std::string>();
    
    for (size_t i=0;i<vad.size();i++) // iterate predictions
      {
	std::string uri = vad.at(i).get("uri").get<std::string>();
	uris.push_back(uri);
	cv::Mat img = imgs.at(i);
	std::vector<APIData> ad_cls = vad.at(i).getv("classes");
	std::vector<APIData> cad_cls;
	
	// rotate and make image available to next service
	if (ad_cls.size() > 0)
	  {
	    std::string cat1 = ad_cls.at(0).get("cat").get<std::string>();
	    cv::Mat rimg, timg;
	    if (cat1 == "0")  // all tests in absolute orientation
	      {
		rimg = img;
	      }
	    else if (cat1 == "90")
	      {
		cv::transpose(img,timg);
		int orient = 1;
		if (orientation == "relative")
		  orient = 0; // 270
		cv::flip(timg,rimg,orient);
	      }
	    else if (cat1 == "180")
	      {
		cv::flip(img,rimg,-1);
	      }
	    else if (cat1 == "270")
	      {
		cv::transpose(img,timg);
		int orient = 0;
		if (orientation == "relative")
		  orient = 1; // 90
		cv::flip(timg,rimg,orient);
	      }
	    if (!rimg.empty())
	      rimgs.push_back(rimg);
	  }
      }
    // store rotated images into action output store
    APIData action_out;
    action_out.add("data_raw_img",rimgs);
    action_out.add("cids",uris);
    cdata.add_action_data(_action_id,action_out);
  }
  
  void ClassFilter::apply(APIData &model_out,
			  ChainData &cdata)
  {
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
	    if ((usit=on_classes_us.find(cat))!=on_classes_us.end())
	      {
		cad_cls.push_back(ad_cls.at(j));
	      }
	  }
	APIData ccls;
	ccls.add("classes",cad_cls);
	if (vad.at(i).has("index_uri"))
	  ccls.add("index_uri",vad.at(i).get("index_uri").get<std::string>());
	ccls.add("uri",vad.at(i).get("uri").get<std::string>());
	cvad.push_back(ccls);
      }

    // empty action data
    cdata.add_action_data(_action_id,APIData());
    //actions_data.push_back(APIData());
    
    // updated model data
    model_out.add("predictions",cvad);
  }

  void ChainActionFactory::apply_action(const std::string &action_type,
                          APIData &model_out,
                          ChainData &cdata) {
    std::string action_id;
    if (_adc.has("id"))
        action_id = _adc.get("id").get<std::string>();
    else action_id = std::to_string(cdata._action_data.size());
    if (action_type == "crop")
      {
        ImgsCropAction act(_adc, action_id, action_type);
        act.apply(model_out, cdata);
      }
    else if (action_type == "rotate")
      {
	ImgsRotateAction act(_adc,action_id,action_type);
	act.apply(model_out,cdata);
      }
    else if (action_type == "filter")
      {
        ClassFilter act(_adc, action_id, action_type);
        act.apply(model_out, cdata);
      }
#ifdef USE_DLIB
    else if (action_type == "dlib_align_crop")
      {
	DlibAlignCropAction act(_adc,action_id,action_type);
	act.apply(model_out,cdata);
      }
#endif
    else {
        throw ActionBadParamException("unknown action " + action_type);
    }
  }

} // end of namespace
