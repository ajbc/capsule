#include "data.h"

Data::Data() {
    max_doc = 0;
    max_term = 0;
    max_entity = 0;
    max_date = 0;
}

void Data::read_training(string counts_filename, string meta_filename) {
    //printf("%s\n", counts_filename.c_str());
    int doc, term, count, author, date;

    // read in metadata
    FILE* fileptr = fopen(meta_filename.c_str(), "r");
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &doc, &author, &date) != EOF)) {
        //printf("reading meta: doc %d, date %d, author %d\n", doc, date, author);
        authors[doc] = author;
        dates[doc] = date;
            
        if (doc > max_doc)
            max_doc = doc;
        if (author > max_entity)
            max_entity = author;
        if (date > max_date)
            max_date = date;
        doc_counts[date]++;
    }
    fclose(fileptr);

    // set up baseline strunctures
    map<int, double> overall_mean;
    map<int, map<int, double> > entity_mean;
    set<int> docset;
    map<int, set<int> > entity_docset;
    map<int, set<int> > day_docset;
    map<int, map<int, set<int> > > entity_day_docset;
    
    // read in training data
    fileptr = fopen(counts_filename.c_str(), "r");
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &doc, &term, &count) != EOF)) {
        //printf("%d\t%d\t%d\n", doc,term,count);
        if (count != 0) {
            train_docs.push_back(doc);
            train_terms.push_back(term);
            train_counts.push_back(count);
            train_set.insert(DocTerm(doc, term));
            if (doc > max_doc)
                max_doc = doc;
            if (term > max_term)
                max_term = term;
            if (doc > max_train_doc)
                max_train_doc = doc;

            overall_mean[term] += 1;
            entity_mean[authors[doc]][term] += 1;
            docset.insert(doc);
            entity_docset[authors[doc]].insert(doc);
            day_docset[dates[doc]].insert(doc);
            entity_day_docset[authors[doc]][dates[doc]].insert(doc);
        }
    }
    fclose(fileptr);

    // for term in terms
    // overall_mean[term] /= docset.size()

    // for each entity
    // for terms used by that entity
    // entity_mean[entity][term] /= entity_docset[entity].size()

    doc_terms = new vector<int>[max_train_doc+1];
    doc_term_counts = new vector<int>[max_train_doc+1];
    for (int i = 0; i < num_training(); i++) {
        doc = train_docs[i];
        term = train_terms[i];
        count = train_counts[i];
        //printf("%d/%d:\t%d/%d\t%d/%d\t%d\n", i,num_training(),doc,max_doc,term, max_term,count);
        doc_terms[doc].push_back(term);
        doc_term_counts[doc].push_back(count);
    }

    for (doc = 0; doc < train_doc_count(); doc++) {
        double dev = 0;
        double ent_dev = 0;
        // look at all the document's terms and caslculate ave deviation form mean
        for (int j = 0; j < term_count(doc); j++) {
            term = get_term(doc, j);
            count = get_term_count(doc, j);
            dev += abs(count - overall_mean[term]/docset.size());
            ent_dev += abs(count - entity_mean[authors[doc]][term]/entity_docset[authors[doc]].size());
        }
        dev /= term_count(doc);
        ent_dev /= term_count(doc);

        if (dev > overall_doc_dist[dates[doc]])
            overall_doc_dist[dates[doc]] = dev;
        if (ent_dev > entity_doc_dist[authors[doc]][dates[doc]])
            entity_doc_dist[authors[doc]][dates[doc]] = ent_dev;
        overall_day_dist[dates[doc]] += dev / day_docset[dates[doc]].size();
        entity_day_dist[authors[doc]][dates[doc]] += ent_dev / entity_day_docset[authors[doc]][dates[doc]].size();
    }


    /*umat locations = umat(2, num_training());
    fcolvec values = fcolvec(num_training());
    user_items = new vector<int>[user_count()];
    for (int i = 0; i < num_training(); i++) {
        locations(0,i) = train_users[i]; // row
        locations(1,i) = train_items[i]; // col
        values(i) = train_ratings[i];
        user_items[train_users[i]].push_back(train_items[i]);
    }
    ratings = sp_fmat(locations, values, user_count(), item_count());

    for (int user = 0; user < user_count(); user++) {
        user_ave_ratings[reverse_user_ids[user]] /= user_items[user].size();
    }
    for (int item = 0; item < item_count(); item++) {
        item_ave_ratings[reverse_item_ids[item]] /= item_popularity[item];
    }*/
}

void Data::read_validation(string filename) {
    // read in validation data
    FILE* fileptr = fopen(filename.c_str(), "r");

    int doc, term, count;
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &doc, &term, &count) != EOF)) {
        if (count != 0) {
            validation_docs.push_back(doc);
            validation_terms.push_back(term);
            validation_counts.push_back(count);
            train_set.insert(DocTerm(doc, term));
        }
    }
    fclose(fileptr);
            
    /*umat locations = umat(2, num_validation());
    fcolvec values = fcolvec(num_validation());
    for (int i = 0; i < num_validation(); i++) {
        locations(0, i) = validation_users[i];
        locations(1, i) = validation_items[i];
        values(i) = validation_ratings[i];
    }

    validation_ratings_matrix = sp_fmat(locations, values, user_count(), item_count());*/
}

void Data::read_test(string filename) {
    // read in test data
    FILE* fileptr = fopen(filename.c_str(), "r");

    int doc, term, count;
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &doc, &term, &count) != EOF)) {
        test_docs.push_back(doc);
        test_terms.push_back(term);
        test_counts.push_back(count);
        test_num_terms[doc]++;
        test_num_docs[term]++;
        test_dat[DocTerm(doc, term)] = count;
    }

    fclose(fileptr);
}

void Data::save_summary(string filename) {
    FILE* file = fopen(filename.c_str(), "w");
    
    fprintf(file, "num documents:\t%d\n", doc_count());
    fprintf(file, "num terms:    \t%d\n", term_count());
    fprintf(file, "num entities: \t%d\n", entity_count());
    fprintf(file, "num dates:    \t%d\n", date_count());
    fprintf(file, "num doc-term counts:\t%d\t%d\t%d\n", num_training(), num_validation(), num_test());
}

int Data::doc_count() {
    return max_doc+1;
}

int Data::doc_count(int date) {
    return doc_counts[date];
}

int Data::train_doc_count() {
    return max_train_doc+1;
}

int Data::term_count() {
    return max_term+1;
}

int Data::entity_count() {
    return max_entity+1;
}

int Data::date_count() {
    return max_date+1;
}

int Data::get_entity(int doc) {
    return authors[doc];
}

int Data::get_date(int doc) {
    return dates[doc];
}

int Data::term_count(int doc) {
    return doc_terms[doc].size();
}

int Data::get_term(int doc, int i) {
    return doc_terms[doc][i];
}

int Data::get_term_count(int doc, int i) {
    return doc_term_counts[doc][i];
}

// training data
int Data::num_training() {
    return train_counts.size();
}

int Data::get_train_doc(int i) {
    return train_docs[i];
}

int Data::get_train_term(int i) {
    return train_terms[i];
}

int Data::get_train_count(int i) {
    return train_counts[i];
}

// validation data
int Data::num_validation() {
    return validation_counts.size();
}

int Data::get_validation_doc(int i) {
    return validation_docs[i];
}

int Data::get_validation_term(int i) {
    return validation_terms[i];
}

int Data::get_validation_count(int i) {
    return validation_counts[i];
}

// test data
int Data::num_test() {
   return test_counts.size();
}

int Data::num_test(int doc) {
    return test_num_terms[doc];
}

int Data::get_test_count(int doc, int term) {
    return test_dat[DocTerm(doc, term)];
}

int Data::num_test_term(int term) {
    return test_num_docs[term];
}


bool Data::in_training(int doc, int term) {
    return train_set.count(DocTerm(doc, term)) != 0;
}

double Data::overall_doc_outlier_dist(int day) {
    return overall_doc_dist[day];
}

double Data::overall_day_ave_dist(int day) {
    return overall_day_dist[day];
}

double Data::entity_doc_outlier_dist(int entity, int day) {
    return entity_doc_dist[entity][day];
}

double Data::entity_day_ave_dist(int entity, int day) {
    return entity_day_dist[entity][day];
}