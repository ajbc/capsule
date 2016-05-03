#include "capsule.h"

Capsule::Capsule(model_settings* model_set, Data* dataset) {
    settings = model_set;
    data = dataset;

    // phi: entity conerns
    printf("\tinitializing entity concerns (phi)\n");
    phi = fmat(settings->k, data->entity_count());
    logphi = fmat(settings->k, data->entity_count());
    a_phi = fmat(settings->k, data->entity_count());
    b_phi = fmat(settings->k, data->entity_count());

    printf("\tinitializing event strengths (psi)\n");
    psi = fvec(data->date_count());
    logpsi = fvec(data->date_count());
    a_psi = fvec(data->date_count());
    b_psi = fvec(data->date_count());

    // theta: doc topics
    printf("\tinitializing doc topics (theta)\n");
    theta = fmat(settings->k, data->doc_count());
    logtheta = fmat(settings->k, data->doc_count());
    a_theta = fmat(settings->k, data->doc_count());
    b_theta = fmat(settings->k, data->doc_count());

    // epsilon: doc events
    printf("\tinitializing doc events (epsilon)\n");
    epsilon = fmat(data->date_count(), data->doc_count());
    logepsilon = fmat(data->date_count(), data->doc_count());
    a_epsilon = fmat(data->date_count(), data->doc_count());
    b_epsilon = fmat(data->date_count(), data->doc_count());

    // beta: global topics
    printf("\tinitializing topics (beta)\n");
    beta = fmat(settings->k, data->term_count());
    logbeta = fmat(settings->k, data->term_count());
    a_beta = fmat(settings->k, data->term_count());
    b_beta = fmat(settings->k, data->term_count());

    // pi: event descriptions
    printf("\tinitializing event descriptions (pi)\n");
    pi = fmat(data->date_count(), data->term_count());
    logpi = fmat(data->date_count(), data->term_count());
    a_pi = fmat(data->date_count(), data->term_count());
    b_pi = fmat(data->date_count(), data->term_count());

    // decay function, for ease
    decay = fmat(data->date_count(), data->date_count());
    logdecay = fmat(data->date_count(), data->date_count());

    // keep track of old a parameters for SVI
    a_phi_old = fmat(settings->k, data->entity_count());
    a_phi_old.fill(settings->a_phi);
    a_psi_old = fvec(data->date_count());
    a_psi_old.fill(settings->a_psi);
    a_theta_old = fmat(settings->k, data->doc_count());
    a_theta_old.fill(settings->a_theta);
    a_epsilon_old = fmat(data->date_count(), data->doc_count());
    a_epsilon_old.fill(settings->a_epsilon);
    a_beta_old = fmat(settings->k, data->term_count());
    a_beta_old.fill(settings->a_beta);
    a_pi_old = fmat(data->date_count(), data->term_count());
    a_pi_old.fill(settings->a_pi);

    printf("\tsetting random seed\n");
    rand_gen = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rand_gen, (long) settings->seed); // init the seed

    initialize_parameters();

    scale = settings->svi ? float(data->train_doc_count()) / float(settings->sample_size) : 1;
}

void Capsule::learn() {
    double old_likelihood, delta_likelihood, likelihood = -1e10;
    int likelihood_decreasing_count = 0;
    time_t start_time, end_time;

    int iteration = 0;
    char iter_as_str[4];
    bool converged = false;
    bool on_final_pass = false;

    while (!converged) {
        time(&start_time);
        iteration++;
        printf("iteration %d\n", iteration);

        reset_helper_params();

        set<int> terms;
        set<int> entities;
        set<int> dates;
        int doc, term, count, entity, date;
        if (settings->svi)
            b_theta.each_col() += sum(beta, 1);

        for (int i = 0; i < settings->sample_size; i++) {
            if (settings->svi) {
                doc = gsl_rng_uniform_int(rand_gen, data->train_doc_count());
            } else {
                doc = i;
            }

            int entity = data->get_entity(doc);
            entities.insert(entity);
            date = data->get_date(doc);
            for (int d = max(0, date - settings->event_dur); d <= date; d++) {
                dates.insert(d);
                b_epsilon(d, doc) += decay(date, d) * accu(pi.row(d));
            }

            // look at all the document's terms
            for (int j = 0; j < data->term_count(doc); j++) {
                term = data->get_term(doc, j);
                terms.insert(term);

                count = data->get_term_count(doc, j);
                update_shape(doc, term, count);
            }

            b_theta.col(doc) += phi.col(entity);
            b_epsilon.col(doc) += psi.col(date);
            if (!settings->svi)
                b_theta.col(doc) += sum(beta, 1);
            update_theta(doc);

            update_epsilon(doc, date);

            for (int k = 0; k < settings->k; k++) {
                a_phi(k, entity) += settings->a_theta * scale;
                b_phi(k, entity) += theta(k, doc) * scale;
            }
        }

        set<int>::iterator it;
        if (!settings->event_only) {
            for (it = entities.begin(); it != entities.end(); it++) {
                entity = *it;
                iter_count_entity[entity]++;
                update_phi(entity);
            }

            b_beta.each_col() += sum(theta, 1);
            update_beta(iteration);
        }

        if (!settings->entity_only) {
            for (it = dates.begin(); it != dates.end(); it++) {
                date = *it;
                iter_count_date[date]++;
                update_psi(date);
                update_pi(date);
            }
        }

        // check for convergence
        if (on_final_pass) {
            printf("Final pass complete\n");
            converged = true;

            old_likelihood = likelihood;
            likelihood = get_ave_log_likelihood();
            delta_likelihood = abs((old_likelihood - likelihood) /
                old_likelihood);
            log_convergence(iteration, likelihood, delta_likelihood);
        } else if (iteration >= settings->max_iter) {
            printf("Reached maximum number of iterations.\n");
            converged = true;

            old_likelihood = likelihood;
            likelihood = get_ave_log_likelihood();
            delta_likelihood = abs((old_likelihood - likelihood) /
                old_likelihood);
            log_convergence(iteration, likelihood, delta_likelihood);
        } else if (iteration % settings->conv_freq == 0) {
            old_likelihood = likelihood;
            likelihood = get_ave_log_likelihood();

            if (likelihood < old_likelihood)
                likelihood_decreasing_count += 1;
            else
                likelihood_decreasing_count = 0;
            delta_likelihood = abs((old_likelihood - likelihood) /
                old_likelihood);
            log_convergence(iteration, likelihood, delta_likelihood);
            if (settings->verbose) {
                printf("delta: %f\n", delta_likelihood);
                printf("old:   %f\n", old_likelihood);
                printf("new:   %f\n", likelihood);
            }
            if (iteration >= settings->min_iter &&
                delta_likelihood < settings->likelihood_delta) {
                printf("Model converged.\n");
                converged = true;
            } else if (iteration >= settings->min_iter &&
                likelihood_decreasing_count >= 2) {
                printf("Likelihood decreasing.\n");
                converged = true;
            }
        }

        // save intermediate results
        if (!converged && settings->save_freq > 0 &&
            iteration % settings->save_freq == 0) {
            printf(" saving\n");
            sprintf(iter_as_str, "%04d", iteration);
            save_parameters(iter_as_str);
        }

        // intermediate evaluation
        if (!converged && settings->eval_freq > 0 &&
            iteration % settings->eval_freq == 0) {
            sprintf(iter_as_str, "%04d", iteration);
            evaluate(iter_as_str);
        }

        time(&end_time);
        log_time(iteration, difftime(end_time, start_time));

        if (converged && !on_final_pass && settings->final_pass) {
            printf("final pass on all users.\n");
            on_final_pass = true;
            converged = false;

            // we need to modify some settings for the final pass
            // things should look exactly like batch for all users
            settings->set_stochastic_inference(false);
            settings->set_sample_size(data->train_doc_count());
            scale = 1;
        }
    }

    save_parameters("final");
}

double Capsule::predict(int doc, int term) {
    double prediction = 0;
    if (!settings->event_only) {
        prediction += accu(theta.col(doc) % beta.col(term));
    }

    if (!settings->entity_only) {
        int date = data->get_date(doc);
        for (int d = max(0, date - settings->event_dur); d <= date; d++)
            prediction += f(date, d) * epsilon(d) * pi(d,term);
    }

    return prediction;
}

// helper function to sort predictions properly
bool prediction_compare(const pair<pair<double,int>, int>& itemA,
    const pair<pair<double, int>, int>& itemB) {
    // if the two values are equal, sort by popularity!
    if (itemA.first.first == itemB.first.first) {
        if (itemA.first.second == itemB.first.second)
            return itemA.second < itemB.second;
        return itemA.first.second > itemB.first.second;
    }
    return itemA.first.first > itemB.first.first;
}

void Capsule::evaluate() {
    evaluate("final", true);
}

void Capsule::evaluate(string label) {
    //evaluate(label, false);
    evaluate(label, true);
}

void Capsule::evaluate(string label, bool write_rankings) {
    time_t start_time, end_time;
    time(&start_time);

    eval(this, &Model::predict, settings->outdir, data, false, settings->seed,
        settings->verbose, label, write_rankings);

    time(&end_time);
    log_time(-1, difftime(end_time, start_time));
}



/* PRIVATE */

void Capsule::initialize_parameters() {
    if (!settings->event_only) {
        // entity concerns
        phi.fill(settings->a_phi / settings->b_phi);
        logphi.fill(gsl_sf_psi(settings->a_phi) - log(settings->b_phi));

        // document topics
        theta.fill(settings->a_theta / (settings->a_phi / settings->b_phi));
        logtheta.fill(gsl_sf_psi(settings->a_theta) - log(settings->a_phi / settings->b_phi));

        // topics
        for (int k = 0; k < settings->k; k++) {
            for (int v = 0; v < data->term_count(); v++) {
                beta(k, v) = (settings->a_beta +
                    gsl_rng_uniform_pos(rand_gen));
                logbeta(k, v) = gsl_sf_psi(beta(k, v));
            }
            logbeta.row(k) -= log(accu(beta.row(k)));
            beta.row(k) /= accu(beta.row(k));
        }
    }

    if (!settings->entity_only) {
        // event strength
        psi.fill(settings->a_psi / settings->b_psi);
        logpsi.fill(gsl_sf_psi(settings->a_psi) - log(settings->b_psi));

        // doc events
        epsilon.fill(settings->a_epsilon / (settings->a_psi / settings->b_psi));
        logepsilon.fill(gsl_sf_psi(settings->a_theta) - log(settings->a_psi / settings->b_psi));

        // event descriptions
        pi.fill(settings->a_pi / 1.0);
        logpi.fill(gsl_sf_psi(settings->a_pi) - log(1.0));
        for (int d = 0; d < data->date_count(); d++) {
            logpi.row(d) -= log(accu(pi.row(d)));
            pi.row(d) /= accu(pi.row(d));
        }

        // log f function
        for (int i = 0; i < data->date_count(); i++) {
            for (int j = 0; j < data->date_count(); j++) {
                // recall: i=docdate, j=eventdate
                decay(i,j) = f(i,j);
                logdecay(i,j) = log(decay(i,j));
            }
        }
    }
}

void Capsule::reset_helper_params() {
    a_phi.fill(settings->a_phi);
    b_phi.fill(settings->b_phi);
    a_psi.fill(settings->a_psi);
    b_psi.fill(settings->b_psi);
    a_theta.fill(settings->a_theta);
    b_theta.fill(0.0);
    a_epsilon.fill(settings->a_epsilon);
    b_epsilon.fill(0.0);
    a_beta.fill(settings->a_beta);
    b_beta.fill(1.0);
    a_pi.fill(settings->a_pi);
    b_pi.fill(1.0);
}

void Capsule::save_parameters(string label) {
    FILE* file;

    if (!settings->event_only) {
        int k;

        // write out phi
        file = fopen((settings->outdir+"/phi-"+label+".dat").c_str(), "w");
        for (int entity = 0; entity < data->entity_count(); entity++) {
            fprintf(file, "%d", entity);
            for (k = 0; k < settings->k; k++)
                fprintf(file, "\t%e", phi(k, entity));
            fprintf(file, "\n");
        }
        fclose(file);

        // write out beta
        file = fopen((settings->outdir+"/beta-"+label+".dat").c_str(), "w");
        for (int term = 0; term < data->term_count(); term++) {
            fprintf(file, "%d", term);
            for (k = 0; k < settings->k; k++)
                fprintf(file, "\t%e", beta(k, term));
            fprintf(file, "\n");
        }
        fclose(file);

        // write out theta
        file = fopen((settings->outdir+"/theta-"+label+".dat").c_str(), "w");
        for (int doc = 0; doc < data->doc_count(); doc++) {
            fprintf(file, "%d", doc);
            for (k = 0; k < settings->k; k++)
                fprintf(file, "\t%e", theta(k, doc));
            fprintf(file, "\n");
        }
        fclose(file);
    }

    if (!settings->entity_only) {
        int t;

        // write out psi
        file = fopen((settings->outdir+"/psi-"+label+".dat").c_str(), "w");
        for (int date = 0; date < data->date_count(); date++)
            fprintf(file, "%e\n", psi(date));
        fclose(file);

        // write out pi
        file = fopen((settings->outdir+"/pi-"+label+".dat").c_str(), "w");
        for (int date = 0; date < data->date_count(); date++) {
            fprintf(file, "%d", date);
            for (t = 0; t < data->term_count(); t++)
                fprintf(file, "\t%e", pi(date, t));
            fprintf(file, "\n");
        }
        fclose(file);

        // write out epsilon
        file = fopen((settings->outdir+"/theta-"+label+".dat").c_str(), "w");
        for (int doc = 0; doc < data->doc_count(); doc++) {
            int date = data->get_date(doc);
            for (int d = max(0, date - settings->event_dur); d <= date; d++)
                fprintf(file, "%d\t%d\t%e\n", doc, d, epsilon(d, doc));
        }
        fclose(file);
    }
}

void Capsule::update_shape(int doc, int term, int count) {
    int date = data->get_date(doc);

    double omega_sum = 0;

    fvec omega_entity, omega_event;
    if (!settings->event_only) {
        omega_entity = exp(logtheta.col(doc) + logbeta.col(term));
        omega_sum += accu(omega_entity);
    }

    if (!settings->entity_only) {
        omega_event = fvec(data->date_count());
        for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
            omega_event(d) = exp(logepsilon(d) + logpi(d,term) + logdecay(date,d));
            omega_sum += omega_event(d);
        }
    }

    if (omega_sum == 0)
        return;

    if (!settings->event_only) {
        omega_entity /= omega_sum * count;
        //for (int k = 0; k < setting->k; k++)
        //    a_phi(k, entity) += pow(theta(k, doc), scale);
        a_theta.col(doc) += omega_entity;
        a_beta.col(term) += omega_entity * scale;
    }

    if (!settings->entity_only) {
        omega_event /= omega_sum * count;
        for (int d = max(0,date - settings->event_dur + 1); d <= date; d++) {
            a_epsilon(d, doc) += omega_event[d];
            a_pi(d, term) += omega_event[d] * scale;
        }
    }
}

void Capsule::update_phi(int entity) {
    if (settings->svi) {
        double rho = pow(iter_count_entity[entity] + settings->delay,
            -1 * settings->forget);
        a_phi.col(entity) = (1 - rho) * a_phi_old.col(entity) + rho * a_phi.col(entity);
        a_phi_old.col(entity) = a_phi.col(entity);
    }

    for (int k = 0; k < settings->k; k++) {
        phi(k, entity) = a_phi(k, entity) / b_phi(k, entity);
        logphi(k, entity) = gsl_sf_psi(a_phi(k, entity)) - log(b_phi(k, entity));
    }
}

void Capsule::update_psi(int date) {
    if (settings->svi) {
        double rho = pow(iter_count_date[date] + settings->delay,
            -1 * settings->forget);
        a_psi.col(date) = (1 - rho) * a_psi_old.col(date) + rho * a_psi.col(date);
        a_psi_old.col(date) = a_psi.col(date);
    }

    psi(date) = a_psi(date) / b_psi(date);
    logpsi(date) = gsl_sf_psi(a_psi(date)) - log(b_psi(date));
}

void Capsule::update_theta(int doc) {
    for (int k = 0; k < settings->k; k++) {
        theta(k, doc) = a_theta(k, doc) / b_theta(k, doc);
        logtheta(k, doc) = gsl_sf_psi(a_theta(k, doc)) - log(b_theta(k, doc));
    }
}

void Capsule::update_epsilon(int doc, int date) {
    for (int d = max(0, date - settings->event_dur); d <= date; d++) {
        epsilon(d, doc) = a_epsilon(d, doc) / b_epsilon(d, doc);
        logepsilon(d, doc) = gsl_sf_psi(a_epsilon(d, doc)) - log(b_epsilon(d, doc));

        a_psi(d) += settings->a_epsilon * scale;
        b_psi(d) += epsilon(d, doc) * scale;

        b_pi.row(d) += decay(date, d) * epsilon(d, doc) * scale;
    }
}

void Capsule::update_beta(int iteration) {
    if (settings->svi) {
        double rho = pow(iteration + settings->delay,
            -1 * settings->forget);
        a_beta = (1 - rho) * a_beta_old + rho * a_beta;
        a_beta_old = a_beta * 1.0;
    }

    for (int k = 0; k < settings->k; k++) {
        for (int v = 0; v < data->term_count(); v++) {
            beta(k, v) = a_beta(k, v) / b_beta(k, v);
            logbeta(k, v) = gsl_sf_psi(a_beta(k, v)) - log(b_beta(k, v));
        }

        logbeta.row(k) -= log(accu(beta.row(k)));
        beta.row(k) /= accu(beta.row(k));
    }
}

void Capsule::update_pi(int date) {
    if (settings->svi) {
        double rho = pow(iter_count_date[date] + settings->delay,
            -1 * settings->forget);
        a_pi.row(date) = (1 - rho) * a_pi_old.row(date) + rho * a_pi.row(date);
        a_pi_old.row(date) = a_beta.row(date);
    }

    for (int v = 0; v < data->term_count(); v++) {
        pi(date, v)  = a_pi(date, v) / b_pi(date, v);
        logpi(date, v) = gsl_sf_psi(a_pi(date, v)) - log(b_pi(date, v));
    }

    logpi.row(date) -= log(accu(pi.row(date)));
    pi.row(date) /= accu(pi.row(date));
}


double Capsule::point_likelihood(double pred, int truth) {
    //return log(pred) * truth - log(factorial(truth)) - pred; (est)
    return log(pred) * truth - pred;
}

double Capsule::get_ave_log_likelihood() {//TODO: rename (it's not ave)
    double prediction, likelihood = 0;
    int doc, term, count;
    for (int i = 0; i < data->num_validation(); i++) {
        doc = data->get_validation_doc(i);
        term = data->get_validation_term(i);
        count = data->get_validation_count(i);

        prediction = predict(doc, term);

        likelihood += point_likelihood(prediction, count);
        //double ll = point_likelihood(prediction, count);
        //likelihood += ll;
        //printf("\t%f\t[%d]\t=> + %f\n", prediction, count, ll);
    }

    printf("likelihood %f\n", likelihood);

    return likelihood;// / data->num_validation();
}

double Capsule::p_gamma(fmat x, fmat a, fmat b) {
    double rv = 0.0;
    for (uint r =0; r < x.n_rows; r++) {
        for (uint c=0; c < x.n_cols; c++) {
            rv += (a(r,c) - 1.0) * log(x(r,c)) - b(r,c) * x(r,c) - a(r,c) * log(b(r,c)) - lgamma(a(r,c));
        }
    }
    return rv;
}

double Capsule::p_gamma(fmat x, double a, fmat b) {
    double rv = 0.0;
    double lga = lgamma(a);
    for (uint r =0; r < x.n_rows; r++) {
        for (uint c=0; c < x.n_cols; c++) {
            rv += (a - 1.0) * log(x(r,c)) - b(r,c) * x(r,c) - a * log(b(r,c)) - lga;
        }
    }
    return rv;
}

double Capsule::p_gamma(fmat x, double a, double b) {
    return accu((a-1) * log(x) - b * x - a * log(b) - lgamma(a));
}

double Capsule::p_gamma(fvec x, fvec a, fvec b) {
    double rv = 0.0;
    for (uint i = 0; i < x.n_elem; i++) {
        rv += (a(i) - 1.0) * log(x(i)) - b(i) * x(i) - a(i) * log(b(i)) - lgamma(a(i));
    }
    return rv;
}

double Capsule::p_gamma(fvec x, double a, fvec b) {
    double rv = 0.0;
    double lga = lgamma(a);
    for (uint i = 0; i < x.n_elem; i++) {
        rv += (a - 1.0) * log(x(i)) - b(i) * x(i) - a * log(b(i)) - lga;
    }
    return rv;
}

double Capsule::p_gamma(fvec x, double a, double b) {
    return accu((a-1) * log(x) - b * x - a * log(b) - lgamma(a));
}

double Capsule::elbo_extra() {
    double rv, rvtotal = 0;

    // subtract q
    if (!settings->entity_only) {
        rv = p_gamma(pi, a_pi, b_pi);
        //printf("%f\t", rv);
        rvtotal -= rv;

        rv = p_gamma(epsilon, a_epsilon, b_epsilon);
        //printf("%f\t", rv);
        rvtotal -= rv;

        rv = p_gamma(psi, a_psi, b_psi);
        //printf("%f\t", rv);
        rvtotal -= rv;
    }

    if (!settings->event_only) {
        rv = p_gamma(beta, a_beta, b_beta);
        //printf("%f\t", rv);
        rvtotal -= rv;

        rv = p_gamma(theta, a_theta, b_theta);
        //printf("%f\t", rv);
        rvtotal -= rv;

        rv = p_gamma(phi, a_phi, b_phi);
        //printf("%f\t", rv);
        rvtotal -= rv;
    }

    // add p
    if (!settings->entity_only) {
        rv = p_gamma(pi, settings->a_pi, 1.0);
        //printf("%f\t", rv);
        rvtotal += rv;

        rv = p_gamma(epsilon, settings->a_epsilon, psi);
        //printf("%f\t", rv);
        rvtotal += rv;

        rv = p_gamma(psi, settings->a_psi, settings->b_psi);
        //printf("%f\t", rv);
        rvtotal += rv;
    }

    if (!settings->event_only) {
        rv = p_gamma(beta, settings->a_beta, 1.0);
        //printf("%f\t", rv);
        rvtotal += rv;

        rv = p_gamma(theta, settings->a_theta, phi);
        //printf("%f\n", rv);
        rvtotal += rv;

        rv = p_gamma(phi, settings->a_phi, settings->b_phi);
        //printf("%f\t", rv);
        rvtotal += rv;
    }

    return rvtotal;
}

void Capsule::log_convergence(int iteration, double ave_ll, double delta_ll) {
    FILE* file = fopen((settings->outdir+"/log_likelihood.dat").c_str(), "a");
    /*if (settings->event_only)
        printf("q evt desc\t\tq evtent\t\tp evt desc\tp evtent\n");
    else if (settings->entity_only)
        printf("q entity\tq topics\t\tp entity\tp topics\n");
    else
        printf("q evt desc\t\tq evtent\t\tq entity\tq topics\tp evt desc\tp evtent\t\tp entity\tp topics\n");*/
    double ee = elbo_extra();
    printf("ll %f\tq/p %f\n", ave_ll, ee);
    fprintf(file, "%d\t%f\t%f\t%f\n", iteration, ave_ll+ee, ave_ll, delta_ll);
    printf("ll: %f\n", ave_ll);
    printf("total: %f\n", ee + ave_ll);

    //fprintf(file, "%d\t%f\t%f\t%f\n", iteration, ave_ll+elbo_extra(), ave_ll, delta_ll);
    fclose(file);
}

void Capsule::log_time(int iteration, double duration) {
    FILE* file = fopen((settings->outdir+"/time_log.dat").c_str(), "a");
    fprintf(file, "%d\t%.f\n", iteration, duration);
    fclose(file);
}

void Capsule::log_user(FILE* file, int user, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg) {
    fprintf(file, "%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user,
        rmse, mae, rank, first, crr, ncrr, ndcg);
}

double Capsule::f(int doc_date, int event_date) {
    // this can be confusing: the document of int
    if (event_date > doc_date || event_date <= (doc_date - settings->event_dur))
        return 0;
    return (1.0-(0.0+doc_date-event_date)/settings->event_dur);
}

double Capsule::get_event_strength(int date) {
    return psi(date);
}
