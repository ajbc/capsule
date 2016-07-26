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
    total_term_count = 0;

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
        doc_counts[make_pair(author, date)]++;
    }
    fclose(fileptr);

    // read in training data
    fileptr = fopen(counts_filename.c_str(), "r");
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &doc, &term, &count) != EOF)) {
        //printf("%d\t%d\t%d\n", doc,term,count);
        if (count != 0) {
            train_docs.push_back(doc);
            train_terms.push_back(term);
            train_counts.push_back(count);
            total_term_count += count;
            vocab_counts[term] += count;
            train_set.insert(DocTerm(doc, term));
            if (doc > max_doc)
                max_doc = doc;
            if (term > max_term)
                max_term = term;
            if (doc > max_train_doc)
                max_train_doc = doc;
        }
    }
    fclose(fileptr);

    doc_terms = new vector<int>[max_train_doc+1];
    doc_term_counts = new vector<int>[max_train_doc+1];
    for (int i = 0; i < num_training(); i++) {
        doc = train_docs[i];
        term = train_terms[i];
        count = train_counts[i];
        doc_terms[doc].push_back(term);
        doc_term_counts[doc].push_back(count);
    }

    for(vector<int>::iterator it = doc_terms->begin(); it != doc_terms->end(); it++) {
        int doc = *it;
        doc_counts_entity[authors[doc]] += 1;
        doc_counts_date[dates[doc]] += 1;
    }
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
}

void Data::read_test(string filename) {
    // read in test data
    FILE* fileptr = fopen(filename.c_str(), "r");

    int doc, term, count;
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &doc, &term, &count) != EOF)) {
        test_docs.push_back(doc);
        test_terms.push_back(term);
        test_counts.push_back(count);
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

int Data::doc_count(int entity, int date) {
    return doc_counts[make_pair(entity, date)];
}

int Data::train_doc_count() {
    return max_train_doc+1;
}

int Data::train_doc_count_by_entity(int entity) {
    return doc_counts_entity[entity];
}

int Data::train_doc_count_by_date(int date) {
    return doc_counts_date[date];
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

int Data::vocab_count(int term) {
    return vocab_counts[term];
}

int Data::total_terms() {
    return total_term_count;
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

int Data::get_test_doc(int i) {
    return test_docs[i];
}

int Data::get_test_term(int i) {
    return test_terms[i];
}

int Data::get_test_count(int i) {
    return test_counts[i];
}
