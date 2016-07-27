#include "capsule.h"
//#include <omp.h>

Capsule::Capsule(model_settings* model_set, Data* dataset) {
    settings = model_set;
    data = dataset;
    last_save = "";

    printf("\tallocating parameters\n");
    if (settings->incl_topics) {
        printf("\t\ttopic parameters\n");
        // beta: global topics
        printf("\t\t\tglobal topics (beta)\n");
        beta = fmat(settings->k, data->term_count());
        logbeta = fmat(settings->k, data->term_count());
        a_beta = fmat(settings->k, data->term_count());
        // keep track of old a parameters for SVI
        a_beta_old = fmat(settings->k, data->term_count());
        a_beta_old.fill(settings->a_beta);

        // phi: entity general concerns
        printf("\t\t\tentity general concerns (phi)\n");
        phi = fmat(settings->k, data->entity_count());
        logphi = fmat(settings->k, data->entity_count());
        a_phi = fmat(settings->k, data->entity_count());
        b_phi = fmat(settings->k, data->entity_count());
        // keep track of old parameters for SVI
        a_phi_old = fmat(settings->k, data->entity_count());
        a_phi_old.fill(settings->a_phi);
        b_phi_old = fmat(settings->k, data->entity_count());
        b_phi_old.fill(settings->b_phi);

        // theta: doc topics
        printf("\t\t\tdoc topics (theta)\n");
        theta = fmat(settings->k, data->doc_count());
        logtheta = fmat(settings->k, data->doc_count());
        a_theta = fmat(settings->k, data->doc_count());
        b_theta = fmat(settings->k, data->doc_count());
    }

    if (settings->incl_events) {
        printf("\t\tevent parameters\n");
        // pi: event descriptions
        printf("\t\t\tevent descriptions (pi)\n");
        pi = fmat(data->date_count(), data->term_count());
        logpi = fmat(data->date_count(), data->term_count());
        a_pi = fmat(data->date_count(), data->term_count());
        // keep track of old a parameters for SVI
        a_pi_old = fmat(data->date_count(), data->term_count());
        a_pi_old.fill(settings->a_pi);

        // psi: event strengths
        printf("\t\t\tevent strengths (psi)\n");
        psi = fvec(data->date_count());
        logpsi = fvec(data->date_count());
        a_psi = fvec(data->date_count());
        b_psi = fvec(data->date_count());
        // keep track of old parameters for SVI
        a_psi_old = fvec(data->date_count());
        a_psi_old.fill(settings->a_psi);
        b_psi_old = fvec(data->date_count());
        b_psi_old.fill(settings->b_psi);

        // epsilon: doc events
        printf("\t\t\tdoc events (epsilon)\n");
        epsilon = sp_fmat(data->date_count(), data->doc_count());
        logepsilon = sp_fmat(data->date_count(), data->doc_count());
        a_epsilon = sp_fmat(data->date_count(), data->doc_count());
        b_epsilon = sp_fmat(data->date_count(), data->doc_count());

        // decay function, for ease
        decay = fmat(data->date_count(), data->date_count());
        logdecay = fmat(data->date_count(), data->date_count());
    }

    if (settings->incl_entity) {
        printf("\t\tentity parameters\n");
        // eta: entity descriptions
        printf("\t\t\tentity descriptions (eta)\n");
        eta = fmat(data->entity_count(), data->term_count());
        logeta = fmat(data->entity_count(), data->term_count());
        a_eta = fmat(data->entity_count(), data->term_count());
        // keep track of old a parameters for SVI
        a_eta_old = fmat(data->entity_count(), data->term_count());
        a_eta_old.fill(settings->a_eta);

        // xi: entity strengths
        printf("\t\t\tentity strengths (xi)\n");
        xi = fvec(data->entity_count());
        logxi = fvec(data->entity_count());
        a_xi = fvec(data->entity_count());
        b_xi = fvec(data->entity_count());
        // keep track of old a parameters for SVI
        a_xi_old = fvec(data->entity_count());
        a_xi_old.fill(settings->a_xi);
        b_xi_old = fvec(data->entity_count());
        b_xi_old.fill(settings->b_xi);

        // zeta: doc entity relvance
        printf("\t\t\tdoc entity relevance (zeta)\n");
        zeta = fvec(data->doc_count());
        logzeta = fvec(data->doc_count());
        a_zeta = fvec(data->doc_count());
        b_zeta = fvec(data->doc_count());
    }

    printf("\tsetting random seed\n");
    rand_gen = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rand_gen, (long) settings->seed); // init the seed

    printf("\tinitializing parameters\n");
    initialize_parameters();

    scale = settings->svi ? float(data->train_doc_count()) / float(settings->sample_size) : 1;
    ent_scale = fvec(data->entity_count());
    evt_scale = fvec(data->date_count());
    if (settings->svi) {
        for (int e = 0; e < data->entity_count(); e++) {
            ent_scale(e) = float(data->train_doc_count_by_entity(e)) / float(settings->sample_size);
        }
        for (int d = 0; d < data->date_count(); d++) {
            evt_scale(d) = float(data->train_doc_count_by_date(d)) / float(settings->sample_size);
        }
    } else {
        ent_scale.fill(1);
        evt_scale.fill(1);
    }
}

void Capsule::learn() {
    double old_likelihood, delta_likelihood, likelihood = -1e10;
    int likelihood_decreasing_count = 0;
    time_t start_time, end_time;

    int iteration = 0;
    char iter_as_str[4];
    bool converged = false;
    bool on_final_pass = false;

    int doc, term, count, entity, date;

    set<int> terms;
    set<int> entities;
    set<int> dates;
    if (!settings->svi) {
        printf("itemizing terms, entities, and dates\n");
        for (term = 0; term < data->term_count(); term++)
            terms.insert(term);
        for (entity = 0; entity < data->entity_count(); entity++)
            entities.insert(entity);
        for (date = 0; date < data->date_count(); date++)
            dates.insert(date);
    }

    while (!converged) {
        time(&start_time);
        iteration++;
        printf("iteration %d\n", iteration);

        reset_helper_params();

        if (settings->svi) {
            terms.clear();
            entities.clear();
            dates.clear();
        }

        time_t sst, st, et;
        time(&sst);
        time(&st);

        for (int i = 0; i < settings->sample_size; i++) {
            if (settings->svi) {
                doc = gsl_rng_uniform_int(rand_gen, data->train_doc_count());
            } else {
                doc = i;
                if (doc > 0 && doc % 10000 == 0) {
                    time(&et);
                    double rmt = (difftime(et, sst) / doc) * (data->doc_count() - doc);
                    printf("\t doc %d / %d\t%ds (est. %f 'til end of iter)\n", doc, data->doc_count(), int(difftime(et, st)), rmt);
                    time(&st);
                }
            }

            int entity = data->get_entity(doc);

            if (settings->svi)
                entities.insert(entity);

            date = data->get_date(doc);
            if (settings->incl_events) {
                for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
                    if (settings->svi) {
                        dates.insert(d);
                        a_epsilon(d, doc) = settings->a_epsilon;
                    }
                    b_epsilon(d, doc) = decay(date, d) * accu(pi.row(d));
                }
            }

            // look at all the document's terms
            for (int j = 0; j < data->term_count(doc); j++) {
                term = data->get_term(doc, j);
                if (settings->svi)
                    terms.insert(term);

                count = data->get_term_count(doc, j);
                update_shape(doc, term, count);
            }

            if (settings->incl_topics) {
                b_theta.col(doc) += phi.col(entity);
                b_theta.col(doc) += sum(beta, 1);
                update_theta(doc);
            }

            if (settings->incl_events) {
                for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
                    b_epsilon(d, doc) += psi(date);
                }
                update_epsilon(doc, date);
            }

            if (settings->incl_entity) {
                b_zeta(doc) = xi(entity) + accu(eta.row(entity));
                update_zeta(doc);
                a_xi(entity) += settings->a_zeta * ent_scale[entity];
                b_xi(entity) += zeta(doc) * ent_scale[entity];
            }

            if (settings->incl_topics) {
                for (int k = 0; k < settings->k; k++) {
                    a_phi(k, entity) += settings->a_theta * scale;
                    b_phi(k, entity) += theta(k, doc) * scale;
                }
            }
        }

        set<int>::iterator it;
        for (it = entities.begin(); it != entities.end(); it++) {
            entity = *it;
            iter_count_entity[entity]++;
            if (settings->incl_topics)
                update_phi(entity);
            if (settings->incl_entity)
                update_xi(entity);
        }

        if (settings->incl_topics)
            update_beta(iteration);

        if (settings->incl_entity)
            update_eta(iteration);

        if (settings->incl_events) {
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
            ent_scale.fill(1);
            evt_scale.fill(1);
        }
    }

    save_parameters("final");
}

double Capsule::predict(int doc, int term) {
    double prediction = 0;

    if (settings->incl_topics)
        prediction += accu(theta.col(doc) % beta.col(term));

    if (settings->incl_entity)
        prediction += zeta(doc) * eta(data->get_entity(doc), term);

    if (settings->incl_events) {
        int date = data->get_date(doc);
        for (int d = max(0, date - settings->event_dur + 1); d <= date; d++)
            prediction += f(date, d) * epsilon(d, doc) * pi(d,term);
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

    // open file for eval
    FILE* file = fopen((settings->outdir+"/eval.dat").c_str(), "a");

    double prediction, likelihood = 0;
    int doc, term, count;
    for (int i = 0; i < data->num_test(); i++) {
        doc = data->get_test_doc(i);
        term = data->get_test_term(i);
        count = data->get_test_count(i);

        prediction = predict(doc, term);

        likelihood += point_likelihood(prediction, count);
    }

    fprintf(file, "held out log likelihood @ %s:\t%e\n", label.c_str(), likelihood);
    fclose(file);

    time(&end_time);
    log_time(-1, difftime(end_time, start_time));
}



/* PRIVATE */

void Capsule::initialize_parameters() {
    if (settings->incl_topics) {
        printf("\t\ttopic parameters\n");
        // entity concerns
        phi.fill(settings->a_phi / settings->b_phi);
        logphi.fill(gsl_sf_psi(settings->a_phi) - log(settings->b_phi));

        // document topics
        theta.fill(settings->a_theta / (settings->a_phi / settings->b_phi));
        logtheta.fill(gsl_sf_psi(settings->a_theta) - log(settings->a_phi / settings->b_phi));

        // topics
        for (int k = 0; k < settings->k; k++) {
            for (int v = 0; v < data->term_count(); v++) {
                if (k == 0) {
                    beta(k, v) = (float)data->term_count(v) / (float)data->total_terms();
                } else {
                    beta(k, v) = (settings->a_beta +
                        gsl_rng_uniform_pos(rand_gen));
                }
                logbeta(k, v) = gsl_sf_psi(beta(k, v));
            }
            logbeta.row(k) -= log(accu(beta.row(k)));
            beta.row(k) /= accu(beta.row(k));
        }
    }

    if (settings->incl_entity) {
        printf("\t\tentity parameters\n");
        // entity strength
        xi.fill(settings->a_xi / settings->b_xi);
        logxi.fill(gsl_sf_psi(settings->a_xi) - log(settings->b_xi));

        // document specific entity strength
        zeta.fill(settings->a_zeta / (settings->a_xi / settings->b_xi));
        logzeta.fill(gsl_sf_psi(settings->a_zeta) - log(settings->a_xi / settings->b_xi));

        // entity descriptions
        for (int i = 0; i < data->entity_count(); i++) {
            for (int v = 0; v < data->term_count(); v++) {
                eta(i, v) = (settings->a_eta +
                    gsl_rng_uniform_pos(rand_gen));
                logeta(i, v) = gsl_sf_psi(eta(i, v));
            }
            logeta.row(i) -= log(accu(eta.row(i)));
            eta.row(i) /= accu(eta.row(i));
        }
    }

    if (settings->incl_events) {
        printf("\t\tevent parameters\n");
        // event strength
        psi.fill(settings->a_psi / settings->b_psi);
        logpsi.fill(gsl_sf_psi(settings->a_psi) - log(settings->b_psi));

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

        // doc events
        int date, v = 0;
        for (int doc = 0; doc < data->doc_count(); doc++) {
            date = data->get_date(doc);
            //v += date - max(0, date - settings->event_dur);
            for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
                v++;
            }
        }
        umat locations = umat(2, v);
        fcolvec values = fcolvec(v);
        v = 0;
        for (int doc = 0; doc < data->doc_count(); doc++) {
            date = data->get_date(doc);
            for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
                locations(0, v) = d;
                locations(1, v) = doc;
                values(v) = 1;
                v++;
            }
        }
        event_cells = sp_fmat(locations, values, data->date_count(), data->doc_count());
        epsilon = event_cells * (settings->a_epsilon / (settings->a_psi / settings->b_psi));
        logepsilon = event_cells * (gsl_sf_psi(settings->a_theta) - log(settings->a_psi / settings->b_psi));
        b_epsilon = event_cells * 1.0; //placeholder to create the correct shape matrix (a_eps is done in reset)
    }
}

void Capsule::reset_helper_params() {
    a_phi.fill(settings->a_phi);
    b_phi.fill(settings->b_phi);
    a_psi.fill(settings->a_psi);
    b_psi.fill(settings->b_psi);
    a_xi.fill(settings->a_xi);
    b_xi.fill(settings->b_xi);
    a_theta.fill(settings->a_theta);
    b_theta.fill(0.0);
    a_zeta.fill(settings->a_zeta);
    b_zeta.fill(0.0);
    a_beta.fill(settings->a_beta);
    a_pi.fill(settings->a_pi);
    a_eta.fill(settings->a_eta);
    if (!settings->svi) {
        a_epsilon = event_cells * settings->a_epsilon;
    }
}

void Capsule::save_parameters(string label) {
    FILE* file;

    if (settings->incl_topics) {
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

        file = fopen((settings->outdir+"/a_beta-"+label+".dat").c_str(), "w");
        for (int term = 0; term < data->term_count(); term++) {
            fprintf(file, "%d", term);
            for (k = 0; k < settings->k; k++)
                fprintf(file, "\t%e", a_beta(k, term));
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

    if (settings->incl_entity) {
        int t;

        // write out xi
        file = fopen((settings->outdir+"/xi-"+label+".dat").c_str(), "w");
        for (int entity = 0; entity < data->entity_count(); entity++)
            fprintf(file, "%e\n", xi(entity));
        fclose(file);

        // write out eta
        file = fopen((settings->outdir+"/eta-"+label+".dat").c_str(), "w");
        for (int entity = 0; entity < data->entity_count(); entity++) {
            fprintf(file, "%d", entity);
            for (t = 0; t < data->term_count(); t++)
                fprintf(file, "\t%e", eta(entity, t));
            fprintf(file, "\n");
        }
        fclose(file);

        file = fopen((settings->outdir+"/a_eta-"+label+".dat").c_str(), "w");
        for (int entity = 0; entity < data->entity_count(); entity++) {
            fprintf(file, "%d", entity);
            for (t = 0; t < data->term_count(); t++)
                fprintf(file, "\t%e", a_eta(entity, t));
            fprintf(file, "\n");
        }
        fclose(file);

        // write out zeta
        file = fopen((settings->outdir+"/zeta-"+label+".dat").c_str(), "w");
        for (int doc = 0; doc < data->doc_count(); doc++) {
            fprintf(file, "%d\t%e\n", doc, zeta(doc));
        }
        fclose(file);
    }

    if (settings->incl_events) {
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

        file = fopen((settings->outdir+"/a_pi-"+label+".dat").c_str(), "w");
        for (int date = 0; date < data->date_count(); date++) {
            fprintf(file, "%d", date);
            for (t = 0; t < data->term_count(); t++)
                fprintf(file, "\t%e", a_pi(date, t));
            fprintf(file, "\n");
        }
        fclose(file);

        // write out epsilon
        file = fopen((settings->outdir+"/epsilon-"+label+".dat").c_str(), "w");
        for (int doc = 0; doc < data->doc_count(); doc++) {
            int date = data->get_date(doc);
            for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
                double val = epsilon(d, doc);
                fprintf(file, "%d\t%d\t%e\t%e\n", doc, d, val, val*decay(date,d));
            }
        }
        fclose(file);
    }

    if (settings->overwrite && last_save != "") {
        if (settings->incl_topics) {
            remove((settings->outdir+"/phi-"+last_save+".dat").c_str());
            remove((settings->outdir+"/beta-"+last_save+".dat").c_str());
            remove((settings->outdir+"/a_beta-"+last_save+".dat").c_str());
            remove((settings->outdir+"/theta-"+last_save+".dat").c_str());
        }

        if (settings->incl_entity) {
            remove((settings->outdir+"/xi-"+last_save+".dat").c_str());
            remove((settings->outdir+"/eta-"+last_save+".dat").c_str());
            remove((settings->outdir+"/a_eta-"+last_save+".dat").c_str());
            remove((settings->outdir+"/zeta-"+last_save+".dat").c_str());
        }

        if (settings->incl_events) {
            remove((settings->outdir+"/psi-"+last_save+".dat").c_str());
            remove((settings->outdir+"/pi-"+last_save+".dat").c_str());
            remove((settings->outdir+"/a_pi-"+last_save+".dat").c_str());
            remove((settings->outdir+"/epsilon-"+last_save+".dat").c_str());
        }
    }
    last_save = label;
}

void Capsule::update_shape(int doc, int term, int count) {
    int date = data->get_date(doc);
    int entity = data->get_entity(doc);

    double omega_sum = 0;

    fvec omega_topics, omega_event;
    double omega_entity = 0;

    if (settings->incl_topics) {
        omega_topics = exp(logtheta.col(doc) + logbeta.col(term));
        omega_sum += accu(omega_topics);
    }

    if (settings->incl_entity) {
        omega_entity = exp(logzeta(doc) + logeta(entity, term));
        omega_sum += omega_entity;
    }

    if (settings->incl_events) {
        omega_event = fvec(data->date_count());
        for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
            omega_event(d) = exp(logepsilon(d) + logpi(d, term) + logdecay(date, d));
            omega_sum += omega_event(d);
        }
    }

    if (omega_sum == 0)
        return;

    if (settings->incl_topics) {
        omega_topics *= count / omega_sum;
        a_theta.col(doc) += omega_topics;
        a_beta.col(term) += omega_topics * scale;
    }

    if (settings->incl_entity) {
        omega_entity *= count / omega_sum;
        a_zeta(doc) += omega_entity;
        a_eta(entity, term) += omega_entity * ent_scale[entity];
    }

    if (settings->incl_events) {
        omega_event *= count / omega_sum;
        for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
            a_epsilon(d, doc) += omega_event[d];
            a_pi(d, term) += omega_event[d] * evt_scale[d];
        }
    }
}

void Capsule::update_phi(int entity) {
    if (settings->svi) {
        double rho = pow(iter_count_entity[entity] + settings->delay,
            -1 * settings->forget);
        a_phi.col(entity) = (1 - rho) * a_phi_old.col(entity) + rho * a_phi.col(entity);
        a_phi_old.col(entity) = a_phi.col(entity);
        b_phi.col(entity) = (1 - rho) * b_phi_old.col(entity) + rho * b_phi.col(entity);
        b_phi_old.col(entity) = b_phi.col(entity);
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
        a_psi(date) = (1 - rho) * a_psi_old(date) + rho * a_psi(date);
        a_psi_old(date) = a_psi(date);
        b_psi(date) = (1 - rho) * b_psi_old(date) + rho * b_psi(date);
        b_psi_old(date) = b_psi(date);
    }

    psi(date) = a_psi(date) / b_psi(date);
    logpsi(date) = gsl_sf_psi(a_psi(date)) - log(b_psi(date));
}

void Capsule::update_xi(int entity) {
    if (settings->svi) {
        double rho = pow(iter_count_entity[entity] + settings->delay,
            -1 * settings->forget);
        a_xi(entity) = (1 - rho) * a_xi_old(entity) + rho * a_xi(entity);
        a_xi_old(entity) = a_xi(entity);
        b_xi(entity) = (1 - rho) * b_xi_old(entity) + rho * b_xi(entity);
        b_xi_old(entity) = b_xi(entity);
    }

    xi(entity) = a_xi(entity) / b_xi(entity);
    logxi(entity) = gsl_sf_psi(a_xi(entity)) - log(b_xi(entity));
}

void Capsule::update_theta(int doc) {
    for (int k = 0; k < settings->k; k++) {
        theta(k, doc) = a_theta(k, doc) / b_theta(k, doc);
        logtheta(k, doc) = gsl_sf_psi(a_theta(k, doc)) - log(b_theta(k, doc));
    }
}

void Capsule::update_zeta(int doc) {
    zeta(doc) = a_zeta(doc) / b_zeta(doc);
    logzeta(doc) = gsl_sf_psi(a_zeta(doc)) - log(b_zeta(doc));
}

void Capsule::update_epsilon(int doc, int date) {
    for (int d = max(0, date - settings->event_dur + 1); d <= date; d++) {
        epsilon(d, doc) = a_epsilon(d, doc) / b_epsilon(d, doc);
        logepsilon(d, doc) = gsl_sf_psi(a_epsilon(d, doc)) - log(b_epsilon(d, doc));

        a_psi(d) += settings->a_epsilon * evt_scale[d];
        b_psi(d) += epsilon(d, doc) * evt_scale[d];
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
            beta(k, v) = a_beta(k, v);
            logbeta(k, v) = gsl_sf_psi(a_beta(k, v));
        }

        logbeta.row(k) -= gsl_sf_psi(accu(beta.row(k)));
        beta.row(k) /= accu(beta.row(k));
    }
}

void Capsule::update_eta(int iteration) {
    if (settings->svi) {
        double rho = pow(iteration + settings->delay,
            -1 * settings->forget);
        a_eta = (1 - rho) * a_eta_old + rho * a_eta;
        a_eta_old = a_eta + 0.0;
    }

    for (int n = 0; n < data->entity_count(); n++) {
        for (int v = 0; v < data->term_count(); v++) {
            eta(n, v) = a_eta(n, v);
            logeta(n, v) = gsl_sf_psi(a_eta(n, v));
        }

        logeta.row(n) -= gsl_sf_psi(accu(eta.row(n)));
        eta.row(n) /= accu(eta.row(n));
    }
}

void Capsule::update_pi(int date) {
    if (settings->svi) {
        double rho = pow(iter_count_date[date] + settings->delay,
            -1 * settings->forget);
        a_pi.row(date) = (1 - rho) * a_pi_old.row(date) + rho * a_pi.row(date);
        a_pi_old.row(date) = a_pi.row(date);
    }

    for (int v = 0; v < data->term_count(); v++) {
        pi(date, v)  = a_pi(date, v);
        logpi(date, v) = gsl_sf_psi(a_pi(date, v));
    }

    logpi.row(date) -= gsl_sf_psi(accu(pi.row(date)));
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
    for (uint r = 0; r < x.n_rows; r++) {
        for (uint c = 0; c < x.n_cols; c++) {
            rv += (a(r,c) - 1.0) * log(x(r,c)) - b(r,c) * x(r,c) - a(r,c) * log(b(r,c)) - lgamma(a(r,c));
        }
    }
    return rv;
}

double Capsule::p_gamma(fmat x, double a, fmat b) {
    double rv = 0.0;
    double lga = lgamma(a);
    for (uint c = 0; c < x.n_cols; c++) {
        int e = data->get_entity(c);
        for (uint r = 0; r < x.n_rows; r++) {
            rv += (a - 1.0) * log(x(r,c)) - b(r,e) * x(r,c) - a * log(b(r,e)) - lga;
        }
    }
    return rv;
}

double Capsule::p_gamma(fmat x, double a, fvec b) {
    double rv = 0.0;
    double lga = lgamma(a);
    for (uint r = 0; r < x.n_rows; r++) {
        for (uint c = 0; c < x.n_cols; c++) {
            rv += (a - 1.0) * log(x(r,c)) - b(r) * x(r,c) - a * log(b(r)) - lga;
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

// special for zeta
double Capsule::p_gammaM(fvec x, double a, fvec b) {
    double rv = 0.0;
    double lga = lgamma(a);
    for (uint i = 0; i < x.n_elem; i++) {
        int m = data->get_entity(i);
        rv += (a - 1.0) * log(x(i)) - b(m) * x(i) - a * log(b(m)) - lga;
    }
    return rv;
}

double Capsule::p_gamma(fvec x, double a, double b) {
    return accu((a-1) * log(x) - b * x - a * log(b) - lgamma(a));
}

double Capsule::p_dir(fmat x, fmat a) {
    double rv = 0.0;
    for (uint r = 0; r < x.n_rows; r++) {
        for (uint c = 0; c < x.n_cols; c++) {
            rv += (a(r,c) - 1.0) * log(x(r,c)) - gsl_sf_lngamma(a(r,c));
        }
        rv += gsl_sf_lngamma(accu(x.row(r)));
    }
    return rv;
}

double Capsule::p_dir(fmat x, double a) {
   double rv = - gsl_sf_lngamma(a) * x.n_rows * x.n_cols;
   for (uint r = 0; r < x.n_rows; r++) {
        for (uint c = 0; c < x.n_cols; c++) {
            rv += (a - 1.0) * log(x(r,c));
        }
        rv += gsl_sf_lngamma(accu(x.row(r)));
    }
    return rv;
}

double Capsule::elbo_extra() {
    double rvtotal = 0;
    printf("start ELBO\n");

    // subtract q
    if (settings->incl_events) {
        rvtotal -= p_dir(pi, a_pi);
        //rvtotal -= p_gamma(epsilon, a_epsilon, b_epsilon);
        rvtotal -= p_gamma(psi, a_psi, b_psi);
    }

    if (settings->incl_entity) {
        rvtotal -= p_dir(eta, a_eta);
        rvtotal -= p_gamma(zeta, a_zeta, b_zeta);
        rvtotal -= p_gamma(xi, a_xi, b_xi);
    }

    if (settings->incl_topics) {
        rvtotal -= p_dir(beta, a_beta);
        rvtotal -= p_gamma(theta, a_theta, b_theta);
        rvtotal -= p_gamma(phi, a_phi, b_phi);
    }

    // add p
    if (settings->incl_events) {
        rvtotal += p_dir(pi, settings->a_pi);
        //rvtotal += p_gamma(epsilon, settings->a_epsilon, psi);
        rvtotal += p_gamma(psi, settings->a_psi, settings->b_psi);
    }

    if (settings->incl_entity) {
        rvtotal += p_dir(eta, settings->a_eta);
        rvtotal += p_gammaM(zeta, settings->a_zeta, xi);
        rvtotal += p_gamma(xi, settings->a_xi, settings->b_xi);
    }

    if (settings->incl_topics) {
        rvtotal += p_dir(beta, settings->a_beta);
        rvtotal += p_gamma(theta, settings->a_theta, phi);
        rvtotal += p_gamma(phi, settings->a_phi, settings->b_phi);
    }
    printf("end ELBO\n");

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
    double ee = 0;//elbo_extra();
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
    if (event_date > doc_date)
        return 0;
    if (settings->event_decay == "step") {
        // dur is the number of active time windows
        if (event_date <= (doc_date - settings->event_dur))
            return 0;
        return 1;
    }
    if (settings->event_decay == "linear") {
        if (event_date <= (doc_date - settings->event_dur))
            return 0;
        return (1.0-(0.0+doc_date-event_date)/settings->event_dur);
    }
    if (settings->event_decay == "exponential") {
        if (doc_date >= event_date + settings->event_dur)
            return 0;
        return exp(- (doc_date - event_date) / (settings->event_dur/5.0));
    }
    return 0; // we should never get here
}

double Capsule::get_event_strength(int date) {
    return psi(date);
}
