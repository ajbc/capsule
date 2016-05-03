#include <iostream>
#define ARMA_64BIT_WORD
#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_psi.h>
#include <list>
//#include <algorithm>

#include "utils.h"
#include "eval.h"

using namespace std;
using namespace arma;
#include "data.h"

struct model_settings {
    bool verbose;

    string outdir;
    string datadir;

    double a_phi;
    double b_phi;
    double a_psi;
    double b_psi;
    double a_theta;
    double b_theta;
    double a_epsilon;
    double b_epsilon;
    double a_pi;
    double a_beta;

    bool entity_only;
    bool event_only;

    int event_dur;

    long   seed;
    int    save_freq;
    int    eval_freq;
    int    conv_freq;
    int    max_iter;
    int    min_iter;
    double likelihood_delta;

    bool   svi;
    bool   final_pass;
    int    sample_size;
    double delay;
    double forget;

    int k;


    void set(bool print, string out, string data, bool use_svi,
             double aphi, double bphi, double apsi, double bpsi,
             double athe, double bthe, double aeps, double beps,
             double api, double abet,
             bool entity, bool event, int dur,
             long rand, int savef, int evalf, int convf,
             int iter_max, int iter_min, double delta,
             bool finalpass,
             int sample, double svi_delay, double svi_forget,
             int num_factors) {
        verbose = print;

        outdir = out;
        datadir = data;

        svi = use_svi;

        a_phi     = aphi;
        b_phi     = bphi;
        a_psi     = apsi;
        b_psi     = bpsi;
        a_theta   = athe;
        b_theta   = bthe;
        a_epsilon = aeps;
        b_epsilon = beps;
        a_pi      = api;
        a_beta      = abet;

        entity_only = entity;
        event_only = event;

        event_dur = dur;

        seed = rand;
        save_freq = savef;
        eval_freq = evalf;
        conv_freq = convf;
        max_iter = iter_max;
        min_iter = iter_min;
        likelihood_delta = delta;

        final_pass = finalpass;
        sample_size = sample;
        delay = svi_delay;
        forget = svi_forget;

        k = num_factors;
    }

    void set_stochastic_inference(bool setting) {
        svi = setting;
    }

    void set_sample_size(int setting) {
        sample_size = setting;
    }

    void save(string filename) {
        FILE* file = fopen(filename.c_str(), "w");

        fprintf(file, "data directory: %s\n", datadir.c_str());

        fprintf(file, "\nmodel specification:\n");
        if (entity_only) {
            fprintf(file, "\tentity factors only\n");
        } else if (event_only) {
            fprintf(file, "\tevent factors only\n");
        } else {
            fprintf(file, "\tfull Capsule model (entity + event factors)\n");
        }

        if (!entity_only)
            fprintf(file, "\nevent duration:\t%d\n", event_dur);

        if (!event_only)
            fprintf(file, "\tK = %d   (number of latent factors for general preferences)\n", k);

        fprintf(file, "\nshape and rate hyperparameters:\n");
        if (!event_only) {
            fprintf(file, "\tphi      (%.2f, %.2f)\n", a_phi, b_phi);
            fprintf(file, "\ttheta    (%.2f, %.2f)\n", a_theta, b_theta);
            fprintf(file, "\tbeta     (%.2f, 1.0)\n", a_beta);
        }
        if (!entity_only) {
            fprintf(file, "\tpsi      (%.2f, %.2f)\n", a_psi, b_psi);
            fprintf(file, "\tepsilon  (%.2f, %.2f)\n", a_epsilon, b_epsilon);
            fprintf(file, "\tpi       (%.2f, 1.0)\n", a_pi);
        }

        fprintf(file, "\ninference parameters:\n");
        fprintf(file, "\tseed:                                     %d\n", (int)seed);
        fprintf(file, "\tsave frequency:                           %d\n", save_freq);
        fprintf(file, "\tevaluation frequency:                     %d\n", eval_freq);
        fprintf(file, "\tconvergence check frequency:              %d\n", conv_freq);
        fprintf(file, "\tmaximum number of iterations:             %d\n", max_iter);
        fprintf(file, "\tminimum number of iterations:             %d\n", min_iter);
        fprintf(file, "\tchange in log likelihood for convergence: %f\n", likelihood_delta);
        fprintf(file, "\tfinal pass after convergence:             %s\n", final_pass ? "yes" : "no");

        if (svi) {
            fprintf(file, "\nStochastic variational inference parameters\n");
            fprintf(file, "\tsample size:                              %d\n", sample_size);
            fprintf(file, "\tSVI delay (tau):                          %f\n", delay);
            fprintf(file, "\tSVI forgetting rate (kappa):              %f\n", forget);
        } else {
            fprintf(file, "\nusing batch variational inference\n");
        }

        fclose(file);
    }
};

//WORKING LINE
class Capsule: protected Model {
    private:
        model_settings* settings;
        Data* data;

        // model parameters
        fmat phi;     // entity concerns (topics/general)
        fvec psi;     // event strengths
        fmat theta;   // doc topics
        fmat epsilon; // doc events
        fmat beta;    // topics
        fmat pi;      // event descriptions
        fmat logphi;  // log variants of above
        fvec logpsi;
        fmat logtheta;
        fmat logepsilon;
        fmat logbeta;
        fmat logpi;

        // helper parameters
        fmat decay;
        fmat logdecay;
        fmat a_phi;
        fmat b_phi;
        fvec a_psi;
        fvec b_psi;
        fmat a_theta;
        fmat b_theta;
        fmat a_epsilon;
        fmat b_epsilon;
        fmat a_beta;
        fmat b_beta;
        fmat a_pi;
        fmat b_pi;
        fmat a_phi_old;
        fvec a_psi_old;
        fmat a_theta_old;
        fmat a_epsilon_old;
        fmat a_beta_old;
        fmat a_pi_old;

        // random number generator
        gsl_rng* rand_gen;

        void initialize_parameters();
        void reset_helper_params();
        void save_parameters(string label);

        // parameter updates
        void update_shape(int doc, int term, int count);
        void update_phi(int entity);
        void update_psi(int date);
        void update_theta(int doc);
        void update_epsilon(int doc, int date);
        void update_beta(int iteration);
        void update_pi(int date);

        double get_ave_log_likelihood();
        double p_gamma(fmat x, fmat a, fmat b);
        double p_gamma(fmat x, double a, fmat b);
        double p_gamma(fmat x, double a, double b);
        double p_gamma(fvec x, fvec a, fvec b);
        double p_gamma(fvec x, double a, fvec b);
        double p_gamma(fvec x, double a, double b);
        double elbo_extra();
        void log_convergence(int iteration, double ave_ll, double delta_ll);
        void log_time(int iteration, double duration);
        void log_params(int iteration, double tau_change, double theta_change);
        void log_user(FILE* file, int user, int heldout, double rmse,
            double mae, double rank, int first, double crr, double ncrr,
            double ndcg);

        // define how to scale updates (training / sample size) (for SVI)
        double scale;

        // counts of number of times an item has been seen in a sample (for SVI)
        map<int,int> iter_count_term;
        map<int,int> iter_count_entity;
        map<int,int> iter_count_date;

        void evaluate(string label);
        void evaluate(string label, bool write_rankings);


    public:
        Capsule(model_settings* model_set, Data* dataset);
        void learn();
        double point_likelihood(double pred, int truth);
        double predict(int user, int item);
        void evaluate();
        double f(int doc_date, int event_date);
        double get_event_strength(int date);

};
