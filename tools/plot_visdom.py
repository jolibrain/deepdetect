import argparse
import visdom
import requests
from dd_client import DD


def get_measures(args):
    dd = DD(args.dd_host, args.dd_port)
    dd.set_return_format(dd.RETURN_PYTHON)
    sinfo = dd.get_train(args.dd_service, measure_hist=True)
    measures = sinfo["body"]["measure_hist"]
    return measures


def plot_measures(args, measures):
    vis = visdom.Visdom(
        server=args.visdom_host,
        port=args.visdom_port,
        env=args.visdom_env,
        raise_exceptions=True,
    )
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
    parser.add_argument("--visdom_host", default="localhost", help="Visdom host")
    parser.add_argument("--visdom_port", type=int, default=8097, help="Visdom port")
    parser.add_argument(
        "--visdom_env", type=str, required=True, help="Visdom destination environment"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    measures = get_measures(args)
    plot_measures(args, measures)
    print(f"http://{args.visdom_host}:{args.visdom_port}/env/{args.visdom_env}")


if __name__ == "__main__":
    main()
