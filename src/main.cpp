#include <getopt.h>
#include "capsule.h"


#include <stdio.h>
//#include <string.h>

//gsl_rng * RANDOM_NUMBER = NULL;

void print_usage_and_exit() {
    // print usage information
    printf("********************** Capsule Event Detection Model **********************\n");
    printf("(c) Copyright 2016 Allison J.B. Chaney  ( achaney@cs.princeton.edu )\n");
    printf("Distributed under MIT License; see LICENSE file for details.\n");

    printf("\nusage:\n");
    printf(" capsule [options]\n");
    printf("  --help            print help information\n");
    printf("  --verbose         print extra information while running\n");

    printf("\n");
    printf("  --out {dir}       save directory, required\n");
    printf("  --data {dir}      data directory, required\n");

    printf("\n");
    printf("  --svi             use stochastic VI (instead of batch VI)\n");
    printf("                    default off for < 10M doc-term counts in training\n");
    printf("  --batch           use batch VI (instead of SVI)\n");
    printf("                    default on for < 10M doc-term counts in training\n");

    printf("\n");
    printf("  --a_phi {a}       shape hyperparamter to phi (entity concerns); default 0.3\n");
    printf("  --b_phi {b}       rate hyperparamter to phi (entity concerns); default 0.3\n");
    printf("  --a_psi {a}       shape hyperparamter to psi (event strength); default 0.3\n");
    printf("  --b_psi {b}       rate hyperparamter to psi (event strength); default 0.3\n");
    printf("  --a_xi {a}        shape hyperparamter to xi (entity strength); default 0.3\n");
    printf("  --b_xi {b}        rate hyperparamter to xi (entity strength); default 0.3\n");
    printf("  --a_theta {a}     shape hyperparamter to theta (doc topics); default 0.3\n");
    printf("  --a_epsilon {a}   shape hyperparamter to epsilon (doc events); default 0.3\n");
    printf("  --a_zeta {a}      shape hyperparamter to epsilon (doc entity relevancy); default 0.3\n");
    printf("  --a_beta {a}      shape hyperparamter to beta (topics); default 0.3\n");
    printf("  --a_pi {a}        shape hyperparamter to pi (event description); default 0.3\n");
    printf("  --a_eta {a}       shape hyperparamter to eta (entity description); default 0.3\n");

    printf("\n");
    printf("  --no_topics       don't consider entity topics aspect of factorization\n");
    printf("  --no_entity       don't consider entity concern aspect of factorization\n");
    printf("  --no_events       don't consider event aspect of factorization\n");

    printf("\n");
    printf("  --event_dur {d}   event duration; default 7\n");
    printf("  --event_decay {d} event decay; options: \"exponential\", \"linear\" (default, triangle), \"none\" (square) \n");

    printf("\n");
    printf("  --seed {seed}     the random seed, default from time\n");
    printf("  --save_freq {f}   the saving frequency, default 20.  Negative value means\n");
    printf("                    no savings for intermediate results.\n");
    printf("  --eval_freq {f}   the intermediate evaluating frequency, default -1.\n");
    printf("                    Negative means no evaluation for intermediate results.\n");
    printf("  --conv_freq {f}   the convergence check frequency, default 10.\n");
    printf("  --max_iter {max}  the max number of iterations, default 300\n");
    printf("  --min_iter {min}  the min number of iterations, default 30\n");
    printf("  --converge {c}    the change in log likelihood required for convergence\n");
    printf("                    default 1e-6\n");
    printf("  --final_pass      do a final pass on all data\n");
    printf("\n");

    printf("  --sample {size}   the stochastic sample size, default 1000\n");
    printf("  --svi_delay {t}   SVI delay >= 0 to down-weight early samples, default 1024\n");
    printf("  --svi_forget {k}  SVI forgetting rate (0.5,1], default 0.75\n");
    printf("\n");

    printf("  --K {K}           the number of topics, default 100\n");

    printf("********************************************************************************\n");

    exit(0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) print_usage_and_exit();

    // variables to store command line args + defaults
    string out = "";
    string data = "";
    bool verbose = false;

    bool svi = false;
    bool batchvi = false;

    double a_phi = 0.3;
    double b_phi = 0.3;
    double a_psi = 0.3;
    double b_psi = 0.3;
    double a_xi = 0.3;
    double b_xi = 0.3;
    double a_theta = 0.3;
    double a_epsilon = 0.3;
    double a_zeta = 0.3;
    double a_beta = 0.3;
    double a_pi = 0.3;
    double a_eta = 0.3;

    // these are really bools, but typed as integers to play nice with getopt
    int incl_topics = 1;
    int incl_entity = 1;
    int incl_events = 1;
    bool final_pass = 0;

    int event_dur = 3;
    string event_decay = "linear";

    time_t t; time(&t);
    long   seed = (long) t;
    int    save_freq = 20;
    int    eval_freq = -1;
    int    conv_freq = 10;
    int    max_iter = 300;
    int    min_iter = 30;
    double converge_delta = 1e-6;

    int    sample_size = 1000;
    double svi_delay = 1024;
    double svi_forget = 0.75;

    int    k = 100;

    // ':' after a character means it takes an argument
    const char* const short_options = "hqo:d:vb1:2:3:4:5:6:7:8:9:0:i:l:r:y:s:w:j:g:x:m:c:a:e:f:pk:";
    const struct option long_options[] = {
        {"help",            no_argument,       NULL, 'h'},
        {"verbose",         no_argument,       NULL, 'q'},
        {"out",             required_argument, NULL, 'o'},
        {"data",            required_argument, NULL, 'd'},
        {"svi",             no_argument, NULL, 'v'},
        {"batch",           no_argument, NULL, 'b'},
        {"a_phi",           required_argument, NULL, '1'},
        {"b_phi",           required_argument, NULL, '2'},
        {"a_psi",           required_argument, NULL, '3'},
        {"b_psi",           required_argument, NULL, '4'},
        {"a_xi",            required_argument, NULL, '5'},
        {"b_xi",            required_argument, NULL, '6'},
        {"a_theta",         required_argument, NULL, '7'},
        {"a_epsilon",       required_argument, NULL, '8'},
        {"a_zeta",          required_argument, NULL, '9'},
        {"a_beta",          required_argument, NULL, '0'},
        {"a_pi",            required_argument, NULL, 'i'},
        {"a_eta",           required_argument, NULL, 'l'},
        {"no_topics",       no_argument, &incl_topics, 0},
        {"no_entity",       no_argument, &incl_entity, 0},
        {"no_events",       no_argument, &incl_events, 0},
        {"event_dur",       required_argument, NULL, 'r'},
        {"event_decay",     required_argument, NULL, 'y'},
        {"seed",            required_argument, NULL, 's'},
        {"save_freq",       required_argument, NULL, 'w'},
        {"eval_freq",       required_argument, NULL, 'j'},
        {"conv_freq",       required_argument, NULL, 'g'},
        {"max_iter",        required_argument, NULL, 'x'},
        {"min_iter",        required_argument, NULL, 'm'},
        {"converge",        required_argument, NULL, 'c'},
        {"sample",          required_argument, NULL, 'a'},
        {"delay",           required_argument, NULL, 'e'},
        {"forget",          required_argument, NULL, 'f'},
        {"final_pass",      no_argument, NULL, 'p'},
        {"K",               required_argument, NULL, 'k'},
        {NULL, 0, NULL, 0}};


    int opt = 0;
    while(true) {
        opt = getopt_long(argc, argv, short_options, long_options, NULL);
        switch(opt) {
            case 'h':
                print_usage_and_exit();
                break;
            case 'q':
                verbose = true;
                break;
            case 'o':
                out = optarg;
                break;
            case 'd':
                data = optarg;
                break;
            case 'v':
                svi = true;
                break;
            case 'b':
                batchvi = true;
                break;
            case '1':
                a_phi = atof(optarg);
                break;
            case '2':
                b_phi = atof(optarg);
                break;
            case '3':
                a_psi = atof(optarg);
                break;
            case '4':
                b_psi = atof(optarg);
                break;
            case '5':
                a_xi = atof(optarg);
                break;
            case '6':
                b_xi = atof(optarg);
                break;
            case '7':
                a_theta = atof(optarg);
                break;
            case '8':
                a_epsilon = atof(optarg);
                break;
            case '9':
                a_zeta = atof(optarg);
                break;
            case '0':
                a_beta = atof(optarg);
                break;
            case 'i':
                a_pi = atof(optarg);
                break;
            case 'l':
                a_eta = atof(optarg);
                break;
            case 'r':
                event_dur = atoi(optarg);
                break;
            case 'y':
                event_decay = optarg;
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'w':
                save_freq = atoi(optarg);
                break;
            case 'j':
                eval_freq = atoi(optarg);
                break;
            case 'g':
                conv_freq = atoi(optarg);
                break;
            case 'x':
                max_iter =  atoi(optarg);
                break;
            case 'm':
                min_iter =  atoi(optarg);
                break;
            case 'c':
                converge_delta =  atoi(optarg);
                break;
            case 'a':
                sample_size = atoi(optarg);
                break;
            case 'e':
                svi_delay = atof(optarg);
                break;
            case 'f':
                svi_forget = atof(optarg);
                break;
            case 'p':
                final_pass = true;
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case -1:
                break;
            case '?':
                print_usage_and_exit();
                break;
            default:
                break;
        }
        if (opt == -1)
            break;
    }

    // print information
    printf("********************************************************************************\n");

    if (out == "") {
        printf("No output directory specified.  Exiting.\n");
        exit(-1);
    }

    if (dir_exists(out)) {
        string rmout = "rm -rf " + out;
        system(rmout.c_str());
    }
    make_directory(out);
    printf("output directory: %s\n", out.c_str());

    if (data == "") {
        printf("No data directory specified.  Exiting.\n");
        exit(-1);
    }

    if (!dir_exists(data)) {
        printf("data directory %s doesn't exist!  Exiting.\n", data.c_str());
        exit(-1);
    }
    printf("data directory: %s\n", data.c_str());

    if (!file_exists(data + "/train.tsv")) {
        printf("training data file (train.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }

    if (!file_exists(data + "/validation.tsv")) {
        printf("validation data file (validation.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }

    if (!incl_topics && !incl_entity && !incl_events) {
        printf("Model must include at least one of: topics, entity, or event factors.  Exiting.\n");
        exit(-1);
    }

    if (svi && batchvi) {
        printf("Inference method cannot be both stochatic (SVI) and batch.  Exiting.\n");
        exit(-1);
    }

    if (batchvi && final_pass) {
        printf("Batch VI doesn't allow for a \"final pass.\" Ignoring this argument.\n");
        final_pass = false;
    }

    printf("\nmodel specification includes:\n");
    if (incl_topics)
        printf("\ttopic factors\n");
    if (incl_entity)
        printf("\tentity factors\n");
    if (incl_events)
        printf("\tevent factors\n");

    if (incl_topics) {
        printf("\tK = %d   (number of latent factors for general preferences)\n", k);
    }

    if (incl_events) {
        printf("\nevent duration: %d\n", event_dur);
        printf("\nevent decay: %s\n", event_decay.c_str());
        if (!(event_decay == "none" || event_decay == "linear" || event_decay == "exponential")) {
            printf("decay shape \"%s\" unknown (options: linear, exponential, none)", event_decay.c_str());
            exit(-1);
        }
    }

    printf("\nshape and rate hyperparameters:\n");
    if (incl_topics) {
        printf("\tphi      (%.2f, %.2f)\n", a_phi, b_phi);
        printf("\ttheta    (%.2f, ---)\n", a_theta);
        printf("\tbeta     (%.2f, 1.0)\n", a_beta);
    }
    if (incl_entity) {
        printf("\txi       (%.2f, %.2f)\n", a_xi, b_xi);
        printf("\tzeta     (%.2f, ---)\n", a_zeta);
        printf("\teta      (%.2f, 1.0)\n", a_eta);
    }
    if (incl_events) {
        printf("\tpsi      (%.2f, %.2f)\n", a_psi, b_psi);
        printf("\tepsilon  (%.2f, ---)\n", a_epsilon);
        printf("\tpi       (%.2f, 1.0)\n", a_pi);
    }

    printf("\ninference parameters:\n");
    printf("\tseed:                                     %d\n", (int)seed);
    printf("\tsave frequency:                           %d\n", save_freq);
    printf("\tevaluation frequency:                     %d\n", eval_freq);
    printf("\tconvergence check frequency:              %d\n", conv_freq);
    printf("\tmaximum number of iterations:             %d\n", max_iter);
    printf("\tminimum number of iterations:             %d\n", min_iter);
    printf("\tchange in log likelihood for convergence: %f\n", converge_delta);
    printf("\tfinal pass after convergence:             %s\n", final_pass ? "yes" : "no");

    if (!batchvi) {
        printf("\nStochastic variational inference parameters\n");
        if (!svi)
            printf("  (may not be used, pending dataset size)\n");
        printf("\tsample size:                              %d\n", sample_size);
        printf("\tSVI delay (tau):                          %f\n", svi_delay);
        printf("\tSVI forgetting rate (kappa):              %f\n", svi_forget);
    } else {
        printf("\nusing batch variational inference\n");
    }


    model_settings settings;
    settings.set(verbose, out, data, svi, a_phi, b_phi, a_psi, b_psi, a_xi, b_xi,
        a_theta, a_epsilon, a_zeta, a_pi, a_beta, a_eta,
        (bool) incl_topics, (bool) incl_entity, (bool) incl_events,
        event_dur, event_decay,
        seed, save_freq, eval_freq, conv_freq, max_iter, min_iter, converge_delta,
        final_pass, sample_size, svi_delay, svi_forget, k);

    // read in the data
    printf("********************************************************************************\n");
    printf("reading data\n");
    Data *dataset = new Data();
    printf("\treading training data\t\t...\t");
    dataset->read_training(settings.datadir + "/train.tsv", settings.datadir + "/meta.tsv");
    printf("done\n");

    printf("\treading validation data\t\t...\t");
    dataset->read_validation(settings.datadir + "/validation.tsv");
    printf("done\n");

    if (!file_exists(data + "/test.tsv")) {
        printf("testing data file (test.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    printf("\treading testing data\t\t...\t");
    dataset->read_test(settings.datadir + "/test.tsv");
    printf("done\n");

    printf("\tsaving data stats\t\t...\t");
    dataset->save_summary(out + "/data_stats.txt");
    printf("done\n");

    // save the run settings
    printf("Saving settings\n");
    if (!svi && !batchvi) {
        if (dataset->num_training() > 10000000) {
            settings.set_stochastic_inference(true);
            printf("using SVI (based on dataset size)\n");
        } else {
            printf("using batch VI (based on dataset size)\n");
        }
    }
    printf("doc count %d\n", dataset->doc_count());
    if (!settings.svi)
        settings.set_sample_size(dataset->doc_count());
    printf("sample size %d\n", settings.sample_size);

    settings.save(out + "/settings.txt");

    // TODO: make this/evaluate below optional (--test_only, --no_test)
    printf("********************************************************************************\n");
    printf("commencing model evaluation\n");

    // create model instance; learn!
    printf("\ncreating model instance\n");
    Capsule *model = new Capsule(&settings, dataset);
    printf("commencing model inference\n");
    model->learn();

    // test the final model fit
    printf("evaluating model on held-out data (TODO)\n");
    model->evaluate();

    delete model;
    delete dataset;

    return 0;
}
