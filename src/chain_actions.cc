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
#include <random>
#include <algorithm>

namespace dd
{

  void ImgsCropAction::apply(APIData &model_out,
			     ChainData &cdata,
			     std::vector<std::string> &meta_uris,
			     std::vector<std::string> &index_uris)
    {
      std::vector<APIData> vad = model_out.getv("predictions");
      std::vector<cv::Mat> imgs = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
      std::vector<std::pair<int,int>> imgs_size = model_out.getobj("input").get("imgs_size").get<std::vector<std::pair<int,int>>>();
      std::vector<std::string> cropped_imgs;
      std::vector<std::string> bbox_ids;
      std::vector<std::string> nmeta_uris;
      
      // check for action parameters
      double bratio = 0.0;
      if (_params.has("padding_ratio"))
	bratio = _params.get("padding_ratio").get<double>(); // e.g. 0.055
      std::vector<APIData> cvad;

      bool save_crops = false;
      if (_params.has("save_crops"))
	save_crops = _params.get("save_crops").get<bool>();

      int random_crops = 0;
      if (_params.has("random_crops"))
	random_crops = _params.get("random_crops").get<int>();

      double min_size_ratio = 1.0;
      if (_params.has("min_size_ratio"))
	min_size_ratio = _params.get("min_size_ratio").get<double>();

      //std::cerr << "min_size_ratio=" << min_size_ratio << std::endl;
      
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

	      // inner random cropping here
	      //TODO: keep orig crop ?
	      if (random_crops)
		{
		  int x_min_side_pixels = std::ceil(cropped_img.cols * min_size_ratio);
		  int y_min_side_pixels = std::ceil(cropped_img.rows * min_size_ratio);

		  //std::cerr << "img cols=" << cropped_img.cols << " / img rows=" << cropped_img.rows << std::endl;
		  //std::cerr << "x_min_side_pixels=" << x_min_side_pixels << " / y_min_side_pixels=" << y_min_side_pixels << std::endl;
		  
		  for (int nc=0;nc<random_crops;nc++)
		    {
		      // generate random bbox coordinates
   		      int rxmin = std::rand() % (cropped_img.cols - x_min_side_pixels); // from 0 to orig_cols-x_min_side_pixels-1
		      int rxmax = std::rand() % (cropped_img.cols - rxmin - x_min_side_pixels) + rxmin + x_min_side_pixels; // from xmin to right of image
		      int rymin = std::rand() % (cropped_img.rows - y_min_side_pixels); // from 0 to max img height - min size
		      int rymax = std::rand() % (cropped_img.rows - rymin - y_min_side_pixels) + rymin + y_min_side_pixels;
		      
		      // crop
		      std::cerr << "rxmin=" << rxmin << " / rymin=" << rymin << " / rxmax=" << rxmax << " / rymax=" << rymax << std::endl;//" / rxmax-rxmin=" << rxmax-rxmin << " / rymax-rymin=" << rymax-rymin << std::endl;
		      cv::Rect roi(rxmin,rymin,rxmax-rxmin,rymax-rymin);
		      //std::cerr << "roi.x=" << roi.x << " / roi.y=" << roi.y << " / roi.width=" << roi.width << " / roi.height=" << roi.height << std::endl;
		      cv::Mat cropped_img_rand = cropped_img(roi);
		      
		      // serialize crop into string (will be auto read by read_element in imginputconn)
		      std::vector<unsigned char> cropped_img_ser;
		      bool str_encoding = cv::imencode(".bmp",cropped_img_rand,cropped_img_ser);
		      if (!str_encoding)
			throw ActionInternalException("crop encoding error for uri " + uri);
		      std::string cropped_img_str = std::string(cropped_img_ser.begin(),cropped_img_ser.end());
		      cropped_imgs.push_back(cropped_img_str);

		      // bbox / would appear into every output branch
		      /*APIData ad_bbox;
		      ad_bbox.add("xmin",rxmin);
		      ad_bbox.add("ymin",rymin);
		      ad_bbox.add("xmax",rxmax);
		      ad_bbox.add("ymax",rymax);
		      ad_cls.at(j).add("random_bbox"+std::to_string(nc),ad_bbox);*/
		      
		      // bbox id
		      std::string bbox_id = genid(uri,"bbox"+std::to_string(j) + "_" + std::to_string(nc));
		      APIData ad_cid;
		      bbox_ids.push_back(bbox_id);
		      ad_cls.at(j).add(bbox_id,ad_cid);

		      nmeta_uris.push_back(meta_uris.at(i));
		    }
		  cad_cls.push_back(ad_cls.at(j));
		}
	      else
		{
		  // adding bbox id
		  std::string bbox_id = genid(uri,"bbox"+std::to_string(j));
		  bbox_ids.push_back(bbox_id);
		  APIData ad_cid;
		  ad_cls.at(j).add(bbox_id,ad_cid);
		  cad_cls.push_back(ad_cls.at(j));
		  
		  // serialize crop into string (will be auto read by read_element in imginputconn)
		  std::vector<unsigned char> cropped_img_ser;
		  bool str_encoding = cv::imencode(".bmp",cropped_img,cropped_img_ser);
		  if (!str_encoding)
		    throw ActionInternalException("crop encoding error for uri " + uri);
		  std::string cropped_img_str = std::string(cropped_img_ser.begin(),cropped_img_ser.end());
		  cropped_imgs.push_back(cropped_img_str);
		  nmeta_uris.push_back(meta_uris.at(i));
		}
	      
	      // save crops if requested
	      if (save_crops)
		{
		  std::string puri = dd_utils::split(uri,'/').back();
		  cv::imwrite("crop_" + puri + "_" + std::to_string(j) + ".png",cropped_img);
		}
	    }
	  APIData ccls;
	  ccls.add("uri",uri);
	  if (vad.at(i).has("index_uri"))
	    ccls.add("index_uri",vad.at(i).get("index_uri").get<std::string>());
	  ccls.add("classes",cad_cls);
	  cvad.push_back(ccls);
	  meta_uris = nmeta_uris;
	}
      
      // store serialized crops into action output store
      APIData action_out;
      action_out.add("data",cropped_imgs);
      action_out.add("cids",bbox_ids);
      cdata.add_action_data(_action_id,action_out);      

      /*std::cerr << "bbox_ids size=" << bbox_ids.size() << std::endl;
	std::cerr << "cropped_imgs size=" << cropped_imgs.size() << std::endl;*/
      
      // updated model data with chain ids
      model_out.add("predictions",cvad);
    }

  //TODO: application to input image directly (before any service)
  void RandomCrops::apply(APIData &model_out,
			  ChainData &cdata,
			  std::vector<std::string> &meta_uris,
			  std::vector<std::string> &index_uris)
  {
    std::vector<APIData> vad = model_out.getv("predictions");
    std::vector<cv::Mat> imgs = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
    std::vector<std::pair<int,int>> imgs_size = model_out.getobj("input").get("imgs_size").get<std::vector<std::pair<int,int>>>();
    std::vector<std::string> cropped_imgs;
    std::vector<std::string> bbox_ids;
    
    // check for action parameters
    int ncrops = 5;
    if (_params.has("ncrops"))
      ncrops = _params.get("ncrops").get<int>();
    double min_size_ratio = 0.1;
    if (_params.has("min_size_ratio"))
      min_size_ratio = _params.get("min_size_ratio").get<double>();

    // random generator / TODO: static ?
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> dist01 {0.0,1.0};
    auto gen01 = [&dist01, &gen]() { return dist01(gen); };

    // only for batch of size 1
    if (vad.size() > 1)
      throw ActionInternalException("random crops only support a single image (batch size of 1)");
    
    // iterate image batch
    std::vector<APIData> cvad;
    for (size_t i=0;i<vad.size();i++)
      {
	std::string uri = vad.at(i).get("uri").get<std::string>();
	
	cv::Mat img = imgs.at(i);
	int orig_cols = imgs_size.at(i).second;
	int orig_rows = imgs_size.at(i).first;

	int x_min_side_pixels = std::ceil(orig_cols * min_size_ratio);
	int y_min_side_pixels = std::ceil(orig_rows * min_size_ratio);
	
	std::vector<APIData> ad_cls = vad.at(i).getv("classes");
	std::vector<APIData> cad_cls;

	for (size_t j=0;j<ad_cls.size();j++)
	  {
	
	    for (int nc=0;nc<ncrops;nc++)
	      {
		// generate random bbox coordinates
		int xmin = std::rand() % (orig_cols - x_min_side_pixels); // from 0 to orig_cols-x_min_side_pixels-1
		int xmax = xmin + std::rand() % (orig_cols - xmin); // from xmin to right of image
		int ymin = std::rand() % (orig_rows - y_min_side_pixels); // from 0 to max img height - min size
		int ymax = ymin + std::rand() % (orig_rows - ymin);
		
		std::cerr << "xmin=" << xmin << " / ymin=" << ymin << " / xmax=" << xmax << " / ymax=" << ymax << std::endl;
		
		// bbox id
		std::string bbox_id = genid(uri,"bbox"+std::to_string(nc));
		bbox_ids.push_back(bbox_id);
		APIData ad_cid;
		ad_cls.at(j).add(bbox_id,ad_cid);
		cad_cls.push_back(ad_cls.at(j));
		
		// crop
		cv::Rect roi(xmin,ymax,xmax-xmin,ymax-ymin);
		cv::Mat cropped_img = img(roi);
		
		// serialize crop into string (will be auto read by read_element in imginputconn)
		std::vector<unsigned char> cropped_img_ser;
		bool str_encoding = cv::imencode(".png",cropped_img,cropped_img_ser);
		if (!str_encoding)
		  throw ActionInternalException("crop encoding error for uri " + uri);
		std::string cropped_img_str = std::string(cropped_img_ser.begin(),cropped_img_ser.end());
		cropped_imgs.push_back(cropped_img_str);
	      }
	  }
	
	APIData ccls;
	ccls.add("uri",uri);
	if (vad.at(i).has("index_uri"))
	  ccls.add("index_uri",vad.at(i).get("index_uri").get<std::string>());
	ccls.add("classes",cad_cls);
	cvad.push_back(ccls);
      }

    // store serialized crops into action output store
    APIData action_out;
    action_out.add("data",cropped_imgs);
    action_out.add("cids",bbox_ids);
    cdata.add_action_data(_action_id,action_out);      
  }

  // XXX: assumes batch size is 1 so all crops belong to the same URI
  void MulticropEnsembling::apply(APIData &model_out,
				  ChainData &cdata,
				  std::vector<std::string> &meta_uris,
				  std::vector<std::string> &index_uris)
  {
    std::vector<APIData> vad = model_out.getv("predictions");
    std::unordered_map<std::string,std::pair<double,int>> multibox_nn;
    std::unordered_map<std::string,std::pair<double,int>>::iterator hit;
    for (size_t i=0;i<vad.size();++i)
      {
	std::vector<APIData> nns_vad = vad.at(i).getv("nns");
	for (size_t j=0;j<nns_vad.size();++j)
	  {
	    std::string nns_uri = nns_vad.at(j).get("uri").get<std::string>();
	    double nns_dist = nns_vad.at(j).get("dist").get<double>();
	    if ((hit=multibox_nn.find(nns_uri))==multibox_nn.end())
	      multibox_nn.insert(std::pair<std::string,std::pair<double,int>>(nns_uri,std::pair<double,int>(nns_dist,1)));
	    else
	      {
		(*hit).second.first += nns_dist;
		(*hit).second.second += 1;
	      }
	  }
      }

    // final ranking
    hit = multibox_nn.begin();
    std::multimap<double,APIData> box_nns;
    while(hit!=multibox_nn.end())
      {
	APIData nn;
	nn.add("uri",(*hit).first);
	double ensembled_dist = (*hit).second.first / static_cast<double>((*hit).second.second);
	nn.add("dist",ensembled_dist);
	box_nns.insert(std::pair<double,APIData>(ensembled_dist,nn));
	++hit;
      }
    std::vector<APIData> ranked_nns;
    for (auto mit=box_nns.begin();mit!=box_nns.end();++mit)
      ranked_nns.push_back((*mit).second);

    model_out.add("global_nns",ranked_nns);
  }
  
  void ClassFilter::apply(APIData &model_out,
			  ChainData &cdata,
			  std::vector<std::string> &meta_uris,
			  std::vector<std::string> &index_uris)
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

} // end of namespace
