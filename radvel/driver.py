"""
Driver functions for the radvel pipeline
"""

import os
import pickle
import copy
import ConfigParser
import pandas as pd

import numpy as np

import radvel

def plots(args):
    for config_file in args.setupfn:
        conf_base = os.path.basename(config_file).split('.')[0]
        statfile = os.path.join(args.outputdir,
                                "{}_radvel.stat".format(conf_base))
        
        
        status = load_status(statfile)

        assert status.getboolean('fit', 'run'), \
          "Must perform max-liklihood fit before plotting"
        post = radvel.posterior.load(status.get('fit', 'postfile'))

        for ptype in args.type:
            print "Creating {} plot for {}".format(ptype, conf_base)
            
            if ptype == 'rv':

                saveto = os.path.join(args.outputdir,
                                      conf_base+'_rv_multipanel.pdf')
                radvel.plotting.rv_multipanel_plot(post, saveplot=saveto,
                                                   **args.plotkw)

            if ptype == 'corner' or ptype == 'trend':
                assert status.getboolean('mcmc', 'run'), \
                "Must run MCMC before making corner or trend plots"

                chains = pd.read_csv(status.get('mcmc', 'chainfile'))

            if ptype == 'corner':
                saveto = os.path.join(args.outputdir, conf_base+'_corner.pdf')
                radvel.plotting.corner_plot(post, chains, saveplot=saveto)

            if ptype == 'trend':
                nwalkers = status.getint('mcmc', 'nwalkers')
                
                saveto = os.path.join(args.outputdir, conf_base+'_trends.pdf')
                radvel.plotting.trend_plot(post, chains, nwalkers, saveto)

            savestate = {'{}_plot'.format(ptype): os.path.abspath(saveto)}
            save_status(statfile, 'plot', savestate)
            
        
def fit(args):

    for config_file in args.setupfn:
        conf_base = os.path.basename(config_file).split('.')[0]
        print "Performing max-likelihood fitting for {}".format(conf_base)

        P, post = radvel.utils.initialize_posterior(config_file)
        post = radvel.fitting.maxlike_fitting(post, verbose=True)

        postfile = os.path.join(args.outputdir,
                                '{}_post_obj.pkl'.format(conf_base))
        post.writeto(postfile)

        savestate = {'run': True,
                     'postfile': os.path.abspath(postfile)}
        save_status(os.path.join(args.outputdir,
                                 '{}_radvel.stat'.format(conf_base)),
                                 'fit', savestate)
            
def mcmc(args):

    for config_file in args.setupfn:
        conf_base = os.path.basename(config_file).split('.')[0]
        statfile = os.path.join(args.outputdir,
                                "{}_radvel.stat".format(conf_base))
        
        status = load_status(statfile)
        
        if status.getboolean('fit', 'run'):
            print "Loading starting positions from previous max-likelihood fit"

            post = radvel.posterior.load(status.get('fit', 'postfile'))
        else:
            P, post = radvel.utils.initialize_posterior(config_file)
            
        
        msg = "Running MCMC for {}, nwalkers = {}, nsteps = {} ...".format(
            conf_base, args.nwalkers, args.nsteps)
        print msg
    
        chains = radvel.mcmc(post,
                             threads=1,
                             nwalkers=args.nwalkers,
                             nrun=args.nsteps)


        # Convert chains into CPS basis
        cpschains = chains.copy()
        for par in post.params.keys():
            if not post.vary[par]:
                cpschains[par] = post.params[par]
                 
        cpschains = post.params.basis.to_cps(cpschains)
        cps_quantile = cpschains.quantile([0.159, 0.5, 0.841])

        # Get quantiles and update posterior object
        post_summary=chains.quantile([0.159, 0.5, 0.841])        

        for k in chains.keys():
            if k in post.params.keys():
                post.params[k] = post_summary[k][0.5]
        
        print "Performing post-MCMC maximum likelihood fit..."
        post = radvel.fitting.maxlike_fitting(post, verbose=False)
        
        cpspost = copy.deepcopy(post)
        cpsparams = post.params.basis.to_cps(post.params)
        cpspost.params.update(cpsparams)
        

        print "Calculating uncertainties..."
        cpspost.uparams = {}
        for par in cpspost.params.keys():
            med = cps_quantile[par][0.5]
            high = cps_quantile[par][0.841] - med
            low = med - cps_quantile[par][0.159]
            err = np.mean([high,low])
            err = radvel.utils.round_sig(err)
            med, err, errhigh = radvel.utils.sigfig(med, err)
            cpspost.uparams[par] = err

        print "Final loglikelihood = %f" % post.logprob()
        print "Final RMS = %f" % post.likelihood.residuals().std()
        print "Best-fit parameters:"
        print cpspost

        print "Saving output files..."
        saveto = os.path.join(args.outputdir, conf_base+'_post_summary.csv')
        post_summary.to_csv(saveto, sep=',')
        
        postfile = os.path.join(args.outputdir,
                                '{}_post_obj.pkl'.format(conf_base))
        post.writeto(postfile)

        csvfn = os.path.join(args.outputdir, conf_base+'_chains.csv.tar.bz2')
        chains.to_csv(csvfn, compression='bz2')


        savestate = {'run': True,
                     'postfile': os.path.abspath(postfile),
                     'chainfile': os.path.abspath(csvfn),
                     'summaryfile': os.path.abspath(saveto),
                     'nwalkers': args.nwalkers,
                     'nsteps': args.nsteps}
        save_status(statfile, 'mcmc', savestate)


def bic(args):
    print "bic {}".format(args.mode)

def tables(args):

    for config_file in args.setupfn:
        conf_base = os.path.basename(config_file).split('.')[0]
        statfile = os.path.join(args.outputdir,
                                "{}_radvel.stat".format(conf_base))
        status = load_status(statfile)

        assert status.getboolean('mcmc', 'run'), \
            "Must run MCMC before making tables"

        P, post = radvel.utils.initialize_posterior(config_file)
        post = radvel.posterior.load(status.get('fit', 'postfile'))
        chains = pd.read_csv(status.get('mcmc', 'chainfile'))
        report = radvel.report.RadvelReport(P, post, chains)

        for tabtype in args.type:
            print "Generating LaTeX code for {} table".format(tabtype)                
            
            if tabtype == 'params':
                tex = report.tabletex(tabtype=tabtype)
                
                saveto = os.path.join(args.outputdir, conf_base+'_params.tex')
                with open(saveto, 'w') as f:
                    print >>f, tex

            if tabtype == 'priors':
                tex = report.tabletex(tabtype=tabtype)
                
                saveto = os.path.join(args.outputdir, conf_base+'_priors.tex')
                with open(saveto, 'w') as f:
                    print >>f, tex
                    
            savestate = {'{}_tex'.format(tabtype): os.path.abspath(saveto)}
            save_status(statfile, 'table', savestate)

                

def physical(args):
    print "multiplying mcmc chains by physical parameters"

def report(args):
    print "Assembling report"

    for config_file in args.setupfn:
        conf_base = os.path.basename(config_file).split('.')[0]
        statfile = os.path.join(args.outputdir,
                                "{}_radvel.stat".format(conf_base))

        status = load_status(statfile)

        P, post = radvel.utils.initialize_posterior(config_file)
        post = radvel.posterior.load(status.get('fit', 'postfile'))
        chains = pd.read_csv(status.get('mcmc', 'chainfile'))
        report = radvel.report.RadvelReport(P, post, chains)

        report_depfiles = []
        for ptype,pfile in status.items('plot'):
            report_depfiles.append(pfile)

        with radvel.utils.working_directory(args.outputdir):
            rfile = os.path.join(conf_base+"_results.pdf")
            report_depfiles = [os.path.basename(p) for p in report_depfiles]
            report.compile(rfile, depfiles=report_depfiles)

    
def save_status(statfile, section, statevars):

    config = ConfigParser.RawConfigParser()
    
    if os.path.isfile(statfile):
        config.read(statfile)

    if not config.has_section(section):
        config.add_section(section)
        
    for key,val in statevars.items():
        config.set(section, key, val)

    with open(statfile, 'w') as f:
        config.write(f)

def load_status(statfile):
    config = ConfigParser.RawConfigParser()
    gl = config.read(statfile)

    return config
