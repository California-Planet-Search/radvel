"""
Command Line Interface
"""
from argparse import ArgumentParser

def fit(args):
    print "Perform max-likelihood fitting"

def mcmc(args):
    print "Perform MCMC exploration"

def bic(args):
    print "bic {}".format(args.mode)

def table_rv(args):
    print "rv table"

def physical(args):
    print "multiplying mcmc chains by physical parameters"

def report(args):
    print "assembling report"

def main():
    psr = ArgumentParser(
        description="RadVel: The Radial Velocity Toolkit", prog='radvel'
    )

    subpsr = psr.add_subparsers(title="subcommands", dest='subcommand')

    # In the parent parser, we define arguments and options common to
    # all subcommands.
    psr_parent = ArgumentParser(add_help=False)
    psr_parent.add_argument(
        '-d', type=str, help="Working directory. Default is ./<starname>/"
    )
    psr_parent.add_argument('-s',
        dest='setupfn', type=str, nargs='+', 
        help="Setup file[s]. Can chain multiple."
    )

    # Fitting    
    psr_fit = subpsr.add_parser(
        'fit', parents=[psr_parent],
        description="Perform max-likelihood fitting"
    )
    psr_fit.set_defaults(func=fit)

    # MCMC
    psr_mcmc = subpsr.add_parser(
        'mcmc', parents=[psr_parent],
        description="Perform MCMC exploration"
    )
    psr_mcmc.add_argument('--nsteps', type=int, help="Number of MCMC steps")
    psr_mcmc.add_argument(
        '--nburn', type=int, help="Number of MCMC burn-in steps"
    )
    psr_mcmc.set_defaults(func=mcmc)


    # Physical parameters
    psr_physical = subpsr.add_parser(
        'physical', parents=[psr_parent],
        description="Multiply MCMC chains by physical parameters. MCMC must"
        + "be run first"
    )

    psr_physical.set_defaults(func=physical)
    
    
    
    # radvel derive epic205071894.py



    # BIC 
    psr_bic = subpsr.add_parser('bic', parents=[psr_parent],)
    psr_bic.add_argument(
        '--mode', type=str, nargs='+', 
        choices=['trend','ecc','nplanets'],
        help="type of BIC comparison to perform"
    )
    psr_bic.set_defaults(func=bic)

    # Tables
    psr_table_parent = ArgumentParser(add_help=False)
    psr_table_parent.add_argument(
        '--header', action='store_true',
        help="included latex column header. Default just prints data rows"
    )
    psr_table = subpsr.add_parser(
        'table', parents=[psr_parent, psr_table_parent], 
        description='Create LaTeX tables'
        
    )
    subpsr_table = psr_table.add_subparsers(
        title="subcommands", dest='subcommand'
    )

    psr_table_rv = subpsr_table.add_parser(
        'rv', description = "Radial velocities"
    )
    psr_table_rv.set_defaults(func=table_rv)

    # Reports
    psr_report = subpsr.add_parser(
        'report', parents=[psr_parent], 
        description="Merge output tables and plots into LaTeX report"
    )
    psr_report.set_defaults(func=report)

    
    args = psr.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
