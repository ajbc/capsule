class Model {
    protected:
        Data* data;
    
    public:
        virtual double predict(int doc, int term) { return 0; };
        virtual double point_likelihood(double pred, int truth) { return 0; };
        virtual double get_event_strength(int day) { return 0; };
};
