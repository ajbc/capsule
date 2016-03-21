#ifndef DATA_H
#define DATA_H


#include <string>
#include <stdio.h>
#include <map>
#include <vector>
#include <set>

#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;

typedef pair<int, int> DocTerm;

class Data {
    private:
        vector<int>* doc_terms;
        vector<int>* doc_term_counts;

        int max_doc;
        int max_train_doc;
        int max_term;
        int max_entity;
        int max_date;
        
        map<int,int> authors;
        map<int,int> dates;
        map<int,int> doc_counts;

        // training data
        vector<int> train_docs;
        vector<int> train_terms;
        vector<int> train_counts;

        // validation data
        vector<int> validation_docs;
        vector<int> validation_terms;
        vector<int> validation_counts;
        //sp_fmat validation_counts_matrix;
       
        // test data - rm?
        set<DocTerm> train_set; 
        map<int, int> test_num_terms;
        map<int, int> test_num_docs;
        // typedef set<Point> List;
        //map<int,int> test_count;
        //map<int,int> test_count_item;

        // simple summaries
        //map<int,float> item_ave_ratings;
        //map<int,float> user_ave_ratings;
        
        // summaires used fror baselines
        map<int, double> overall_doc_dist;
        map<int, double> overall_day_dist;
        map<int, map<int, double> > entity_doc_dist;
        map<int, map<int, double> > entity_day_dist;
    
    public:
        //sp_fmat ratings;
        //sp_fmat network_spmat;
        
        Data();
        void read_training(string counts_filename, string meta_filename);
        void read_validation(string filename);
        //WORKING LINE
        void read_test(string filename);
        void save_summary(string filename);

        int doc_count();
        int doc_count(int date);
        int train_doc_count();
        int term_count();
        int entity_count();
        int date_count();

        int term_count(int doc);
        int get_term(int doc, int i);
        int get_term_count(int doc, int i);

        // metadata associated with each document
        int get_entity(int doc);
        int get_date(int doc);

        // training data
        int num_training();
        int get_train_doc(int i);
        int get_train_term(int i);
        int get_train_count(int i);
        
        // validation data
        int num_validation();
        int get_validation_doc(int i);
        int get_validation_term(int i);
        int get_validation_count(int i);
        
        // test data
        vector<int> test_docs;
        vector<int> test_terms;
        vector<int> test_counts;
        
        map<DocTerm,int> test_dat; 
        int num_test();
        int num_test(int doc);
        int get_test_count(int doc, int term);
        int num_test_term(int term);
        /*set<int> test_users;
        set<int> test_items;
        sp_umat test_ratings;
        int num_test_item(int item);*/
        bool in_training(int doc, int term);

        // outlier distances
        double overall_doc_outlier_dist(int day);
        double overall_day_ave_dist(int day);
        double entity_doc_outlier_dist(int entity, int day);
        double entity_day_ave_dist(int entity, int day);
};

#endif
