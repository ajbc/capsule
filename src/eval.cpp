#include "eval.h"

// random generator to break ties
gsl_rng* rand_gen = gsl_rng_alloc(gsl_rng_taus);

// helper function to write out per-doc info
void log_doc(FILE* file, Data *data, int doc, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg, bool stats) {
    if (stats)
        fprintf(file, "%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", doc, 
            heldout, data->term_count(doc), 
            rmse, mae, rank, first, crr, ncrr, ndcg);
    else
        fprintf(file, "%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", doc,
            rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}

void log_term(FILE* file, Data *data, int term, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg, bool stats) {
    if (stats)
        fprintf(file, "%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", term, heldout,
            rmse, mae, rank, first, crr, ncrr, ndcg);
    else
        fprintf(file, "%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", term, 
            rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}

// helper function to sort predictions properly
bool prediction_compare(const pair<double,int>& termA, 
    const pair<double, int>& termB) {
    // if the two values are equal, sort by popularity!
    if (termA.first == termB.first) {
        return gsl_rng_uniform_int(rand_gen, 2) != 0;
    }
    return termA.first > termB.first;
}


// take a prediction function as an argument
void eval(Model* model, double (Model::*prediction)(int,int), string outdir, Data* data, bool stats, 
    unsigned long int seed, bool verbose, string label, bool write_rankings) { 
    // random generator to break ties
    gsl_rng_set(rand_gen, seed);
    
    // test the final model fit
    printf("evaluating model on held-out data\n");
    
    FILE* file = fopen((outdir+"/rankings_" + label + ".tsv").c_str(), "w");
    if (write_rankings)
        fprintf(file, "doc.id\tterm.id\tpred\trank\tcount\n");
    
    FILE* doc_file = fopen((outdir+"/doc_eval_" + label + ".tsv").c_str(), "w");
    if (stats)
        fprintf(doc_file, "doc.id\tnum.heldout\tnum.train\tdegree\tconnectivity\trmse\tmae\tave.rank\tfirst\tcrr\tncrr\tndcg\n");
    else
        fprintf(doc_file, "doc.id\trmse\tmae\tave.rank\tfirst\tcrr\tncrr\tndcg\n");
     
    FILE* term_file = fopen((outdir+"/term_eval_" + label + ".tsv").c_str(), "w");
    fprintf(term_file, "term.id\tpopularity\theldout\trmse\tmae\tave.rank\tfirst\tcrr\tncrr\tndcg\n");
    
    // overall metrics to track
    double rmse = 0;
    double mae = 0;
    double aggr_rank = 0;
    double crr = 0;
    double doc_sum_rmse = 0;
    double doc_sum_mae = 0;
    double doc_sum_rank = 0;
    double doc_sum_first = 0;
    double doc_sum_crr = 0;
    double doc_sum_ncrr = 0;
    double doc_sum_ndcg = 0;

    // per doc attibutes
    double doc_rmse = 0;
    double doc_mae = 0;
    int doc_heldout = 0;
    double doc_rank = 0;
    int first = 0;
    double doc_crr = 0;
    double doc_ncrr = 0;
    double doc_ncrr_normalizer = 0;
    double doc_ndcg = 0;
    double doc_ndcg_normalizer = 0;

    // helper var for evaluation (used for mulitple metrics)
    double local_metric;

    // helper var to hold predicted count
    double pred;
        
    // overall attributes to track
    int doc_count = 0;
    int heldout_count = 0;
    
    int doc, term, count, rank;
    list<pair<double, int> > counts;
    int total_pred = 0;
   
    for (vector<int>::iterator iter_doc = data->test_docs.begin(); 
        iter_doc != data->test_docs.end();
        iter_doc++){

        doc = *iter_doc;
        if (verbose) {
            printf("doc %d\n", doc);
        }
        doc_count++;

        doc_rmse = 0;
        doc_mae = 0;
        doc_rank = 0;
        first = 0;
        doc_crr = 0;
        doc_ncrr_normalizer = 0;
        doc_ndcg = 0;
        doc_ndcg_normalizer = 0;
        doc_heldout = 0;

        for (vector<int>::iterator iter_term = data->test_terms.begin(); 
            iter_term != data->test_terms.end();
            iter_term++){

            term = *iter_term;

            // don't rank terms that we've already seen
            if (data->in_training(doc, term))
                continue;

            total_pred++;

            double p = 0;
            p = (model->*prediction)(doc, term);

            counts.push_back(make_pair(p, term));
        }
        
        counts.sort(prediction_compare);

        rank = 0;
        int test_count = data->num_test(doc);
        while (doc_heldout < test_count && !counts.empty()) {
            pair<double, int> pred_set = counts.front();
            term = pred_set.second;
            count = data->get_test_count(doc, term);
            pred = pred_set.first;
            rank++;
            if (rank <= 1000 && write_rankings) { // TODO: make this threshold a command line arg
                fprintf(file, "%d\t%d\t%f\t%d\t%d\n", doc,
                    term, pred, rank, count);
            }

            // compute metrics only on held-out terms
            if (count != 0) {
                doc_heldout++;
                heldout_count++;

                local_metric = pow(count - pred, 2);
                rmse += local_metric;
                doc_rmse += local_metric;
                
                local_metric = abs(count - pred);
                mae += local_metric;
                doc_mae += local_metric;

                aggr_rank += rank;
                doc_rank += rank;

                local_metric = 1.0 / rank;
                doc_crr += local_metric;
                crr += local_metric;
                doc_ncrr_normalizer += 1.0 / doc_heldout;

                doc_ndcg += count / log(rank + 1);
                doc_ndcg_normalizer += count / log(doc_heldout + 1);

                if (first == 0)
                    first = rank;
            }
            
            counts.pop_front();
        }
        while (!counts.empty()){
            counts.pop_front();
        }

        // log this doc's metrics
        doc_rmse = sqrt(doc_rmse / doc_heldout);
        doc_mae /= doc_heldout;
        doc_rank /= doc_heldout;
        doc_ncrr = doc_crr / doc_ncrr_normalizer;
        doc_ndcg /= doc_ndcg_normalizer;
        
        log_doc(doc_file, data, doc, doc_heldout, doc_rmse, 
            doc_mae, doc_rank, first, doc_crr, doc_ncrr, doc_ndcg, stats);

        // add this doc's metrics to overall metrics
        doc_sum_rmse += doc_rmse;
        doc_sum_mae += doc_mae;
        doc_sum_rank += doc_rank;
        doc_sum_first += first;
        doc_sum_crr += doc_crr;
        doc_sum_ncrr += doc_ncrr;
        doc_sum_ndcg += doc_ndcg;
    }
    fclose(doc_file);
    fclose(file);
    if (!write_rankings)
        remove((outdir+"/rankings_" + label + ".tsv").c_str());

    
    // per term attibutes
    double term_rmse = 0;
    double term_mae = 0;
    int term_heldout = 0;
    double term_rank = 0;
    double term_crr = 0;
    double term_ncrr = 0;
    double term_ncrr_normalizer = 0;
    double term_ndcg = 0;
    double term_ndcg_normalizer = 0;

    for (vector<int>::iterator iter_term = data->test_terms.begin(); 
        iter_term != data->test_terms.end();
        iter_term++){

        term = *iter_term;
        if (verbose) {
            printf("term %d\n", term);
        }

        term_rmse = 0;
        term_mae = 0;
        term_rank = 0;
        first = 0;
        term_crr = 0;
        term_ncrr_normalizer = 0;
        term_ndcg = 0;
        term_ndcg_normalizer = 0;
        term_heldout = 0;

        for (vector<int>::iterator iter_doc = data->test_docs.begin(); 
            iter_doc != data->test_docs.end();
            iter_doc++){

            doc = *iter_doc;

            // don't rank terms that we've already seen
            if (data->in_training(doc, term))
                continue;

            total_pred++;

            counts.push_back(make_pair((model->*prediction)(doc, term), doc));
        }
        
        counts.sort(prediction_compare);

        rank = 0;
        int test_count = data->num_test_term(term);
        while (term_heldout < test_count && !counts.empty()) {
            pair<double, int> pred_set = counts.front();
            doc = pred_set.second;
            count = data->get_test_count(doc, term);
            pred = pred_set.first;
            rank++;

            // compute metrics only on held-out terms
            if (count != 0) {
                term_heldout++;

                term_rmse += pow(count - pred, 2);
                term_mae += abs(count - pred);
                term_rank += rank;
                term_crr += 1.0 / rank;
                term_ncrr_normalizer += 1.0 / term_heldout;

                term_ndcg += count / log(rank + 1);
                term_ndcg_normalizer += count / log(term_heldout + 1);

                if (first == 0)
                    first = rank;
            }
            
            counts.pop_front();
        }
        while (!counts.empty()){
            counts.pop_front();
        }

        // log this term's metrics
        term_rmse = sqrt(term_rmse / term_heldout);
        term_mae /= term_heldout;
        term_rank /= term_heldout;
        term_ncrr = term_crr / term_ncrr_normalizer;
        term_ndcg /= term_ndcg_normalizer;
        
        log_term(term_file, data, term, term_heldout, term_rmse, 
            term_mae, term_rank, first, term_crr, term_ncrr, term_ndcg, stats);
    }
    fclose(term_file);
    
    // write out results
    file = fopen((outdir+"/eval_summary_" + label + ".dat").c_str(), "w");
    fprintf(file, "metric\tdoc average\theldout pair average\n");
    fprintf(file, "RMSE\t%f\t%f\n", doc_sum_rmse/doc_count, 
        sqrt(rmse/heldout_count));
    fprintf(file, "MAE\t%f\t%f\n", doc_sum_mae/doc_count, mae/heldout_count);
    fprintf(file, "rank\t%f\t%f\n", doc_sum_rank/doc_count, 
        aggr_rank/heldout_count);
    fprintf(file, "first\t%f\t---\n", doc_sum_first/doc_count);
    fprintf(file, "CRR\t%f\t%f\n", doc_sum_crr/doc_count, crr/heldout_count);
    fprintf(file, "NCRR\t%f\t---\n", doc_sum_ncrr/doc_count);
    fprintf(file, "NDCG\t%f\t---\n", doc_sum_ndcg/doc_count);
    fclose(file);
}
