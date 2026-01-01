import argparse as ap
import shutil
import logging
import os

from .component import Component
from .reaction import Reaction
from .chemeqsystem import ChemEqSystem
from sandlerprops.properties import get_database

banner = """
sandlerchemeq -- chemical equilibrium calculations via Gibbs energy minimization

(c) 2025, Cameron F. Abrams <cfa22@drexel.edu>
"""

logger = logging.getLogger(__name__)

def setup_logging(args):    
    loglevel_numeric = getattr(logging, args.logging_level.upper())
    if args.log:
        if os.path.exists(args.log):
            shutil.copyfile(args.log, args.log+'.bak')
        logging.basicConfig(filename=args.log,
                            filemode='w',
                            format='%(asctime)s %(name)s %(message)s',
                            level=loglevel_numeric
        )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s> %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def solve(args):
    db = get_database()
    components = []
    for i, cname in enumerate(args.components):
        c = Component.from_compound(db.get_compound(cname), T=args.temperature, P=args.pressure)
        components.append(c)
    reactions = []
    for reaction_str in args.reactions:
        component_names = reaction_str.split(',')
        reaction_components = []
        for cname in component_names:
            if 'hydrogen' in cname:
                cname = 'hydrogen (equilib)'
            c = Component.from_compound(db.get_compound(cname), T=args.temperature, P=args.pressure)
            reaction_components.append(c)
        rxn = Reaction(components=reaction_components)
        reactions.append(rxn)
    system = ChemEqSystem(Components=components, Reactions=reactions,
                          N0=args.initial_moles,
                          T=args.temperature,
                          P=args.pressure)
    if system.M == 0:
        system.solve_lagrange()
    else:
        system.solve_implicit(Xinit=args.X_init)
    print(system.report())
    # print("Equilibrium moles of each component:")
    # for i, c in enumerate(system.Components):
    #     print(f"  {c.Name}: N={system.N[i]:.6f} y={system.ys[i]:.6f}")

def enforce_schema(args):
    # user must specify either components or reactions
    if not args.components and not args.reactions:
        raise ValueError("You must specify at least one component via --components or at least one reaction via --reactions")
    if args.components and args.reactions:
        raise ValueError("You may only specify either components via --components or reactions via --reactions, not both")
    if not args.components and args.reactions:
        # parse reactions to determine components
        component_list = []
        for reaction_str in args.reactions:
            components_in_reaction = reaction_str.split(',')
            for comp in components_in_reaction:
                if 'hydrogen' in comp:
                    comp = 'hydrogen (equilib)'
                component_list.append(comp)
        args.components = component_list
    if args.components and not args.reactions:
        args.reactions = []
        c = []
        for cname in args.components:
            if 'hydrogen' in cname:
                cname = 'hydrogen (equilib)'
            c.append(cname)
        args.components = c
    return args

def cli():
    subcommands = {
        'solve': dict(
            func = solve,
            help = 'Solve chemical equilibrium'
        ),
    }
    parser = ap.ArgumentParser(
        prog='sandlerchemeq',
        description='Chemical equilibrium calculations via Gibbs energy minimization'
    )
    parser.add_argument(
        '-b',
        '--banner',
        default=False,
        action=ap.BooleanOptionalAction,
        help='toggle banner message'
    )
    parser.add_argument(
        '--logging-level',
        type=str,
        default='debug',
        choices=[None, 'info', 'debug', 'warning'],
        help='Logging level for messages written to diagnostic log'
    )
    parser.add_argument(
        '-l',
        '--log',
        type=str,
        default='',
        help='File to which diagnostic log messages are written'
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        metavar="<command>",
        required=True,
    )
    command_parsers={}
    for k, specs in subcommands.items():
        command_parsers[k] = subparsers.add_parser(
            k,
            help=specs['help'],
            formatter_class=ap.RawDescriptionHelpFormatter
        )
        command_parsers[k].set_defaults(func=specs['func'])
    
    command_parsers['solve'].add_argument(
        '-c',
        '--components',
        type=str,
        nargs='+',
        required=False,
        help='List of component names'
    )

    command_parsers['solve'].add_argument(
        '-T',
        '--temperature',
        type=float,
        required=True,
        help='Temperature in Kelvin'
    )

    command_parsers['solve'].add_argument(
        '-P',
        '--pressure',
        type=float,
        required=True,
        help='Pressure in MPa'
    )

    command_parsers['solve'].add_argument(
        '-n0',
        '--initial-moles',
        type=float,
        nargs='+',
        required=True,
        help='Initial moles of each component'
    )

    command_parsers['solve'].add_argument(
        '-r',
        '--reactions',
        type=str,
        nargs='*',
        default=[],
        help='List of reactions, each as a comma-delimited string of component names (no spaces)'
    )

    command_parsers['solve'].add_argument(
        '-xinit',
        '--X-init',
        type=float,
        nargs='*',
        default=None,
        help='Initial guess for reaction extents (if not provided, defaults to zero for all reactions)'
    )

    args = parser.parse_args()
    args = enforce_schema(args)
    setup_logging(args)

    if args.banner:
        print(banner)
    if hasattr(args, 'func'):
        args.func(args)
    else:
        my_list = ', '.join(list(subcommands.keys()))
        print(f'No subcommand found. Expected one of {my_list}')
    if args.banner:
        print('Thanks for using sandlerchemeq!')