import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cfg", "--config", type=str, required=True, help="Path to the config file"
)
parser.add_argument("--msg", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from utility.make_env import make_env
from utility.parameter import Parameter
from utility.setup_logger import setup_logging
from utility.alg_init import alg_init
import envs


def main():
    parameter = Parameter(args_cli.config, arg_cli=args_cli)
    parameter.log_file_path = setup_logging(parameter)
    env = make_env(parameter)
    alg = alg_init(parameter, env)
    alg.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
