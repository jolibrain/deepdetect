import argparse
import time

import requests
import visdom

from deepdetect.cli.confusion import write_confusion_table
from deepdetect.cli.events import MetricEventExtractor


def plot_measures(vis, measures):
    for title, y in measures.items():
        title = title.removesuffix("_hist")
        x = list(range(len(y)))
        vis.line(X=x, Y=y, win=title, opts={"title": title})


def parse_args():
    parser = argparse.ArgumentParser(
        description="plot DeepDetect train measures to a Visdom server"
    )
    parser.add_argument("--dd_host", default="localhost", help="DeepDetect server host")
    parser.add_argument(
        "--dd_port", type=int, default=8080, help="DeepDetect server port"
    )
    parser.add_argument(
        "--dd_service", required=True, help="DeepDetect train service to plot"
    )
    parser.add_argument("--visdom_host", default="http://localhost", help="Visdom host")
    parser.add_argument("--visdom_port", type=int, default=8097, help="Visdom port")
    parser.add_argument(
        "--visdom_env", type=str, required=True, help="Visdom destination environment"
    )
    parser.add_argument("--visdom-base-url", default="/", help="Visdom server base path")
    parser.add_argument("--job", type=int, default=1)
    parser.add_argument("--follow", action="store_true", help="poll until the training job finishes")
    return parser.parse_args()


def main():
    args = parse_args()
    endpoint = f"http://{args.dd_host}:{args.dd_port}/train"
    vis = visdom.Visdom(server=args.visdom_host, port=args.visdom_port, env=args.visdom_env,
                        base_url=args.visdom_base_url, raise_exceptions=True)
    extractor = MetricEventExtractor()
    while True:
        response = requests.get(endpoint, params={
            "service": args.dd_service, "job": args.job,
            "parameters.output.measure_hist": "true",
        }, timeout=30)
        response.raise_for_status()
        status = response.json()
        body = status["body"]
        plot_measures(vis, body.get("measure_hist", {}))
        for key, matrix in extractor.new_confusion_matrices(body):
            write_confusion_table(vis, key, matrix)
        if not args.follow or status["head"]["status"] != "running":
            break
        time.sleep(1)
    print(f"{vis.server}:{vis.port}{vis.base_url}/env/{args.visdom_env}")


if __name__ == "__main__":
    main()
