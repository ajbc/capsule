class Model {
    protected:
        Data* data;
    
    public:
        virtual double predict(int user, int item) { return 0; };
        virtual double point_likelihood(double pred, int truth) { return 0; };
};
