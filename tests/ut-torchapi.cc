/**
 * DeepDetect
 * Copyright (c) 2019 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
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

#include "deepdetect.h"
#include "jsonapi.h"
#include "txtinputfileconn.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string incept_repo = "../examples/torch/resnet50_torch/";
static std::string bert_classif_repo = "../examples/torch/bert_inference_torch/";

TEST(torchapi, service_predict)
{
    // create service
    JsonAPI japi;
    std::string sname = "imgserv";
    std::string jstr = "{\"mllib\":\"torch\",\"description\":\"resnet-50\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  incept_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":224,\"width\":224,\"rgb\":true,\"std\":255.0}}}";
    std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
    ASSERT_EQ(created_str,joutstr);

    // predict
    std::string jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,\"width\":224,\"rgb\":true,\"std\":255.0},\"output\":{\"best\":1}},\"data\":[\"" + incept_repo + "cat.jpg\"]}";
    joutstr = japi.jrender(japi.service_predict(jpredictstr));
    JDoc jd;
    std::cout << "joutstr=" << joutstr << std::endl;
    jd.Parse(joutstr.c_str());
    ASSERT_TRUE(!jd.HasParseError());
    ASSERT_EQ(200,jd["status"]["code"]);
    ASSERT_TRUE(jd["body"]["predictions"].IsArray());
    std::string cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
    ASSERT_TRUE(cl1 == "n02123045 tabby, tabby cat");
    ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0.3);
}

TEST(torchapi, service_predict_txt_classification)
{
    // create service
    JsonAPI japi;
    std::string sname = "txtserv";
    std::string jstr = "{\"mllib\":\"torch\",\"description\":\"bert\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + bert_classif_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\",\"ordered_words\":true,"
        "\"wordpiece_tokens\":true,\"punctuation_tokens\":true,\"width\":512}}}";
    std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
    ASSERT_EQ(created_str,joutstr);

    // predict
    std::string jpredictstr = "{\"service\":\"txtserv\",\"parameters\":{\"output\":{\"best\":1}},\"data\":["
        "\"Get the official USA poly ringtone or colour flag on your mobile for tonights game! Text TONE or FLAG to 84199. Optout txt ENG STOP Box39822 W111WX £1.50\"]}";
    joutstr = japi.jrender(japi.service_predict(jpredictstr));
    JDoc jd;
    std::cout << "joutstr=" << joutstr << std::endl;
    jd.Parse(joutstr.c_str());
    ASSERT_TRUE(!jd.HasParseError());
    ASSERT_EQ(200, jd["status"]["code"]);
    ASSERT_TRUE(jd["body"]["predictions"].IsArray());
    std::string cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
    ASSERT_TRUE(cl1 == "spam");
    ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0.7);
}

TEST(inputconn, txt_tokenize_ordered_words) {
    std::string str = "everything runs fine, right?";
    TxtInputFileConn tifc;
    tifc._ordered_words = true;
    tifc._wordpiece_tokens = true;
    tifc._punctuation_tokens = true;

    tifc._vocab["every"] = Word();
    tifc._vocab["##ing"] = Word();
    tifc._vocab["##thing"] = Word();
    tifc._vocab["fine"] = Word();
    tifc._vocab[","] = Word();
    tifc._vocab["?"] = Word();
    tifc._vocab["right"] = Word();

    tifc.parse_content(str,1);
    TxtOrderedWordsEntry &towe = *dynamic_cast<TxtOrderedWordsEntry*>(tifc._txt.at(0));
    std::vector<std::string> tokens{"every", "##thing", "[UNK]", "fine", ",", "right", "?"};
    ASSERT_EQ(tokens, towe._v);
}
