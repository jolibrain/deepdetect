# DeepDetect benchmark tool
#
# Licence:
# Copyright (c) 2017 Emmanuel Benazara
# Copyright (c) 2020 Jolibrain
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# """
#
# """
# This is a benchmark for image services with DeepDetect.
#
# Instructions:
# - wget https://deepdetect.com/stuff/bench.tar.gz
# - untar bench.tar.gz onto the machine where your DeepDetect server runs
# - remotely benchmark your server with (adapt options as needed):
#
#   dd_bench --host yourhost --port 8080 --sname faces --img-width 300 --img-height 300 --gpu --remote-bench-data-dir /server/path/to/bench/ --max-batch-size 64

import argparse
import csv
import json

from dd_client import DD


def main():
    parser = argparse.ArgumentParser(description="DeepDetect benchmark tool")
    parser.add_argument("--host", help="server host", default="localhost")
    parser.add_argument("--port", help="server port", type=int, default=8080)
    parser.add_argument("--sname", help="service name")
    parser.add_argument("--img-width", help="image width", type=int, default=224)
    parser.add_argument("--img-height", help="image height", type=int, default=224)
    parser.add_argument("--bw", help="whether images are bw", action="store_true")
    parser.add_argument("--rgb", help="whether images are rgb", action="store_true")
    parser.add_argument(
        "--scale",
        help="Multiply all pixels value by a constant factor",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--histogram-equalization",
        "--eqhist",
        help="whether we apply an histogram equalization to images",
        action="store_true",
    )
    parser.add_argument("--gpu", help="whether to bench GPU", action="store_true")
    parser.add_argument("--gpuid", help="gpu id to use", type=int, default=0)
    parser.add_argument(
        "--remote-bench-data-dir",
        help="when bench data directory, when available remotely on the server",
    )
    parser.add_argument(
        "--max-batch-size", help="max batch size to be tested", type=int, default=256
    )
    parser.add_argument(
        "--max-workspace-size",
        help="max workspace size for tensort bench",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--list-bench-files",
        help="file holding the list of bench files",
        default="list_bench_files.txt",
    )
    parser.add_argument(
        "--npasses", help="number of passes for every batch size", type=int, default=5
    )
    parser.add_argument(
        "--detection", help="whether benching a detection model", action="store_true"
    )
    parser.add_argument(
        "--segmentation",
        help="whether benching a segmentation model",
        action="store_true",
    )
    parser.add_argument(
        "--regression",
        help="whether benching a regression model",
        action="store_true",
    )
    parser.add_argument(
        "--search",
        help="whether benching a similarity search service",
        action="store_true",
    )
    parser.add_argument(
        "--search-multibox",
        help="whether benching a multibox similarity search service",
        action="store_true",
    )
    parser.add_argument("--create", help="model's folder name to create a service")
    parser.add_argument(
        "--nclasses",
        help="number of classes for service creation",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--auto-kill",
        help="auto kill the service after benchmarking",
        action="store_true",
    )
    parser.add_argument("--csv-output", help="CSV file output")
    parser.add_argument("--json-output", help="JSON file output")
    parser.add_argument(
        "--mllib", help="mllib to bench, ie [tensorrt|ncnn|caffe]", default="caffe"
    )
    parser.add_argument(
        "--datatype", help="datatype for tensorrt [fp16|fp32]", default="fp32"
    )
    parser.add_argument(
        "--recreate",
        help="recreate service between every batchsize, useful for batch_size dependent precompiling backends (ie tensorRT)",
        action="store_true",
        default=False,
    )
    parser.add_argument("--dla", help="use dla", action="store_true", default=False)
    parser.add_argument(
        "--gpu-resize", help="image resizing on gpu", action="store_true", default=False
    )
    parser.add_argument(
        "--image-interp",
        help="image interpolation method (nearest, linear, cubic, ...)",
    )
    args = parser.parse_args()

    host = args.host
    port = args.port
    dd = DD(host, port)
    dd.set_return_format(dd.RETURN_PYTHON)
    autokill = args.auto_kill

    def service_create(bs):
        # Create a service
        if args.create:
            description = "image classification service"
            mllib = args.mllib
            model = {"repository": args.create}
            parameters_input = {
                "connector": "image",
                "width": args.img_width,
                "height": args.img_height,
                "bw": args.bw,
                "rgb": args.rgb,
                "histogram_equalization": args.histogram_equalization,
                "scale": args.scale,
            }
            if args.segmentation:
                parameters_input["segmentation"] = True
            if args.regression:
                parameters_input["regression"] = True
            if args.dla:
                parameters_mllib = {
                    "nclasses": args.nclasses,
                    "datatype": args.datatype,
                    "readEngine": True,
                    "writeEngine": True,
                    "maxBatchSize": bs,
                    "dla": 0,
                    "maxWorkspaceSize": args.max_workspace_size,
                    "gpu": args.gpu,
                    "gpuid": args.gpuid,
                }
            else:
                parameters_mllib = {
                    "nclasses": args.nclasses,
                    "datatype": args.datatype,
                    "readEngine": True,
                    "writeEngine": True,
                    "maxBatchSize": bs,
                    "maxWorkspaceSize": args.max_workspace_size,
                    "gpu": args.gpu,
                    "gpuid": args.gpuid,
                }
            parameters_output = {}
            dd.put_service(
                args.sname,
                model,
                description,
                mllib,
                parameters_input,
                parameters_mllib,
                parameters_output,
            )
        else:
            pass

    out_json = []
    out_csv = None
    csv_writer = None
    if args.csv_output:
        out_csv = open(args.csv_output, "w+")
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow(["batch_size", "mean processing time", "mean time per img"])

    list_bench_files = []
    with open(args.list_bench_files) as f:
        for line in f:
            list_bench_files.append(args.remote_bench_data_dir + "/" + line.rstrip())
    batch_sizes = []
    batch_size = 1
    while batch_size <= args.max_batch_size:
        batch_sizes.append(batch_size)
        if batch_size < 32:
            batch_size = batch_size * 2
        else:
            batch_size += 16

    parameters_input = {"width": args.img_width, "height": args.img_height}
    if not args.image_interp == "":
        parameters_input["interp"] = args.image_interp
    if args.gpu_resize:
        parameters_input["cuda"] = args.gpu_resize
    parameters_mllib = {"gpu": args.gpu, "gpuid": args.gpuid}
    parameters_output = {}
    if args.detection:
        parameters_output["confidence_threshold"] = 0.1
        if args.search or args.search_multibox:
            parameters_output["search"] = True
            parameters_output["rois"] = "rois"
            parameters_output["bbox"] = False
        else:
            parameters_output["bbox"] = True
        if args.search_multibox:
            parameters_output["multibox_rois"] = True
    elif args.segmentation:
        parameters_input["segmentation"] = True
    elif args.regression:
        parameters_output["regression"] = True
    elif args.search:
        parameters_output["search"] = True

    # First call to load model
    data = list_bench_files[:1]
    if not args.recreate:
        if not args.mllib == "tensorrt" or args.recreate:
            service_create(1)
        else:
            service_create(args.max_batch_size)
        classif = dd.post_predict(
            args.sname, data, parameters_input, parameters_mllib, parameters_output
        )

    for b in batch_sizes:
        data = list_bench_files[:b]
        fail = False
        if args.recreate:
            service_create(b)
            for i in range(5):
                classif = dd.post_predict(
                    args.sname,
                    data,
                    parameters_input,
                    parameters_mllib,
                    parameters_output,
                )
        mean_ptime = 0
        mean_ptime_per_img = 0
        start_transform_time = -1
        for i in range(0, args.npasses + 1):
            print("testing batch size = %s" % len(data))
            classif = dd.post_predict(
                args.sname, data, parameters_input, parameters_mllib, parameters_output
            )
            if classif["status"]["code"] == 200:
                if i == 0:
                    # initialize total transform time
                    sinfo = dd.get_service(args.sname)
                    start_transform_time = sinfo["body"]["service_stats"][
                        "total_transform_duration_ms"
                    ]
                    continue  # skipping first pass so that the batch resize does not affect timing
                ptime = classif["head"]["time"]
                ptime_per_img = ptime / b
                mean_ptime += ptime
                mean_ptime_per_img += ptime_per_img
                print(
                    "pass %s batch size = %s / processing time = %s / time per image = %s"
                    % (i, b, ptime, ptime_per_img)
                )
            else:
                print(classif["status"])
                # reload model
                data = list_bench_files[:1]
                classif = dd.post_predict(
                    args.sname,
                    data,
                    parameters_input,
                    parameters_mllib,
                    parameters_output,
                )
                fail = True
                break
        sinfo = dd.get_service(args.sname)
        mean_transform_time = (
            sinfo["body"]["service_stats"]["total_transform_duration_ms"]
            - start_transform_time
        ) / args.npasses
        mean_processing_time = mean_ptime / args.npasses
        mean_time_per_img = mean_ptime_per_img / args.npasses
        print(
            ">>> batch size = %s / mean processing time = %s / mean time per image = %s / mean transform time = %s / mean transform time per image = %s / fps = %s / fail = %s"
            % (
                b,
                mean_ptime / args.npasses,
                mean_ptime_per_img / args.npasses,
                mean_transform_time,
                mean_transform_time / b,
                1000 / (mean_ptime_per_img / args.npasses),
                fail,
            ),
        )
        out_json.append(
            {
                "batch_size": b,
                "mean_processing_time": mean_processing_time,
                "mean_time_per_img": mean_time_per_img,
            }
        )
        if args.csv_output:
            csv_writer.writerow([b, mean_processing_time, mean_time_per_img])
        # break
        if args.recreate:
            dd.delete_service(args.sname)

    if args.json_output:
        with open(args.json_output, "w") as outfile:
            json.dump(out_json, outfile)

    if autokill:
        dd.delete_service(args.sname)


if __name__ == "__main__":
    main()
