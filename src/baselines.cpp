#include <getopt.h>
#include <stdio.h>
#include <list>
#include "utils.h"
#include "data.h"
#include "eval.h"

void print_usage_and_exit() {
    // print usage information
    printf("************************ Predict Events using Baselines *************************\n");
    printf("(c) Copyright 2016 Allison J.B. Chaney  ( achaney@cs.princeton.edu )\n");
    printf("Distributed under MIT License; see LICENSE file for details.\n");
    
    printf("\nusage:\n");
    printf(" ./pop [options]\n");
    printf("  --help            print help information\n");
    printf("  --verbose         print extra information while running\n");

    printf("\n");
    printf("  --out {dir}       save directory, required\n");
    printf("  --data {dir}      data directory, required\n");
    printf("  --seed {seed}     the random seed, default from time\n");
    
    printf("*********************************************************************************\n");

    exit(0);
}

// helper function to write out per-user info
/*void log_item(FILE* file, Data *data, int item, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg) {
    fprintf(file, "%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", item, 
        data->item_id(item), data->popularity(item), heldout,
        rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}*/

class DocOutlier: protected Model {
    public:
        double predict(int doc, int term) {
            return 0;//data->overall_doc_outlier_dist();
        }

        void evaluate(Data* d, string outdir, bool verbose, long seed) {
            data = d;
            eval(this, &Model::predict, outdir, data, true, seed, verbose, "final", true);
        }
        
        double get_event_strength(int day) {
            //printf("\t\tday %d\n", day);
            return data->overall_doc_outlier_dist(day);
        }
};

class DayOutlier: protected Model {
    public:
        double predict(int doc, int term) {
            return 0;//data->overall_doc_outlier_dist();
        }

        void evaluate(Data* d, string outdir, bool verbose, long seed) {
            data = d;
            eval(this, &Model::predict, outdir, data, true, seed, verbose, "final", true);
        }
        
        double get_event_strength(int day) {
            return data->overall_day_ave_dist(day);
        }
};

class EntityDocOutlier: protected Model {
    public:
        double predict(int doc, int term) {
            return 0;//data->overall_doc_outlier_dist();
        }

        void evaluate(Data* d, string outdir, bool verbose, long seed) {
            data = d;
            eval(this, &Model::predict, outdir, data, true, seed, verbose, "final", true);
        }
        
        double get_event_strength(int day) {
            double rv = 0;
            for (int entity = 0; entity < data->entity_count(); entity++) {
                if (data->entity_doc_outlier_dist(entity, day) > rv)
                    rv = data->entity_doc_outlier_dist(entity, day);
            }
            return rv;
        }
};

class EntityDayOutlier: protected Model {
    public:
        double predict(int doc, int term) {
            return 0;//data->overall_doc_outlier_dist();
        }

        void evaluate(Data* d, string outdir, bool verbose, long seed) {
            data = d;
            eval(this, &Model::predict, outdir, data, true, seed, verbose, "final", true);
        }
        
        double get_event_strength(int day) {
            double rv = 0;
            for (int entity = 0; entity < data->entity_count(); entity++) {
                if (data->entity_day_ave_dist(entity, day) > rv)
                    rv = data->entity_day_ave_dist(entity, day);
            }
            return rv;
        }
};

int main(int argc, char* argv[]) {
    if (argc < 2) print_usage_and_exit();

    // variables to store command line args + defaults
    string outdir = "";
    string datadir = "";
    bool verbose = false;
    long seed = 11;

    // ':' after a character means it takes an argument
    const char* const short_options = "hqo:d:s:";
    const struct option long_options[] = {
        {"help",            no_argument,       NULL, 'h'},
        {"verbose",         no_argument,       NULL, 'q'},
        {"out",             required_argument, NULL, 'o'},
        {"data",            required_argument, NULL, 'd'},
        {"seed",            required_argument, NULL, 's'},
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
                outdir = optarg;
                break;
            case 'd':
                datadir = optarg;
                break;
            case 's':
                seed = atoi(optarg);
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

    if (outdir == "") {
        printf("No output directory specified.  Exiting.\n");
        exit(-1);
    }
    
    if (dir_exists(outdir)) {
        string rmout = "rm -rf " + outdir;
        system(rmout.c_str());
    }
    make_directory(outdir);
    printf("output directory: %s\n", outdir.c_str());
    
    if (datadir == "") {
        printf("No data directory specified.  Exiting.\n");
        exit(-1);
    }
    
    if (!dir_exists(datadir)) {
        printf("data directory %s doesn't exist!  Exiting.\n", datadir.c_str());
        exit(-1);
    }
    printf("data directory: %s\n", datadir.c_str());

    if (!file_exists(datadir + "/train.tsv")) {
        printf("training data file (train.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    
    if (!file_exists(datadir + "/validation.tsv")) {
        printf("validation data file (validation.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    

    // read in the data
    printf("********************************************************************************\n");
    printf("reading data\n");
    Data *dataset = new Data();
    printf("\treading training data\t\t...\t");
    dataset->read_training(datadir + "/train.tsv", datadir + "/meta.tsv");
    printf("done\n");

    printf("\treading validation data\t\t...\t");
    dataset->read_validation(datadir + "/validation.tsv");
    printf("done\n");
    
    if (!file_exists(datadir + "/test.tsv")) {
        printf("testing data file (test.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    printf("\treading testing data\t\t...\t");
    dataset->read_test(datadir + "/test.tsv");
    printf("done\n");
    
    //printf("\tsaving data stats\t\t...\t");
    //data->save_summary(outdir + "/data_stats.txt");
    //printf("done\n");
    
    printf("********************************************************************************\n");
    printf("commencing model evaluation\n");
    
    // test the final model fit
    EntityDayOutlier model1 = EntityDayOutlier();
    EntityDocOutlier model2 = EntityDocOutlier();
    DayOutlier model3 = DayOutlier();
    DocOutlier model4 = DocOutlier();
    printf("\ttotal doc\n");
    make_directory(outdir+ "/totaldoc");
    model4.evaluate(dataset, outdir + "/totaldoc", verbose, seed);
    printf("\ttotal day\n");
    make_directory(outdir+ "/totalday");
    model3.evaluate(dataset, outdir + "/totalday", verbose, seed);
    printf("\tentity doc\n");
    make_directory(outdir+ "/entitydoc");
    model2.evaluate(dataset, outdir + "/entitydoc", verbose, seed);
    printf("\tentity day\n");
    make_directory(outdir+ "/entityday");
    model1.evaluate(dataset, outdir + "/entityday", verbose, seed);

    delete dataset;
    
    return 0;
}
