#include "Halide.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "schema.h"


using namespace Halide;
using namespace Halide::Internal;
using std::vector;
using std::string;
using std::unordered_map;
using Halide::Derivative;


// Convert a vector of Vars to Exprs. Useful for generating references
// to Funcs.
vector<Expr> make_arguments(vector<Var> vars) {
    vector<Expr> result;
    for (Var i : vars) {
        result.push_back(i);
    }
    return result;
}

std::mt19937 rng;

// Helpers to generate random values.
int rand_int(int min, int max) { return (rng() % (max - min + 1)) + min; }
bool rand_bool() { return rng() % 2 == 0; }
float rand_float() { return rand_int(0, 1 << 30) / (float)(1 << 30); }

// Generate random expressions. Given a vector of expresions and a
// tree depth, recursively generates an expression by combining
// subexpressions.  At the base case where depth is 0, we just return
// a randomly chosen input.
Type expr_types[] = { UInt(8), UInt(16), UInt(32), Int(8), Int(16), Int(32), Float(32) };
const int expr_type_count = sizeof(expr_types)/sizeof(expr_types[0]);

typedef Expr (*make_bin_op_fn)(Expr, Expr);

make_bin_op_fn make_bin_op[] = {
    (make_bin_op_fn)operator+,
    (make_bin_op_fn)operator-,
    (make_bin_op_fn)operator*,
    (make_bin_op_fn)min,
    (make_bin_op_fn)max,
    (make_bin_op_fn)operator/,
    (make_bin_op_fn)operator%,
};

make_bin_op_fn make_bool_bin_op[] = {
    (make_bin_op_fn)operator&&,
    (make_bin_op_fn)operator||,
};

make_bin_op_fn make_comp_bin_op[] = {
    (make_bin_op_fn)operator==,
    (make_bin_op_fn)operator!=,
    (make_bin_op_fn)operator<,
    (make_bin_op_fn)operator<=,
    (make_bin_op_fn)operator>,
    (make_bin_op_fn)operator>=
};

const int bin_op_count = sizeof(make_bin_op) / sizeof(make_bin_op[0]);
const int bool_bin_op_count = sizeof(make_bool_bin_op) / sizeof(make_bool_bin_op[0]);
const int comp_bin_op_count = sizeof(make_comp_bin_op) / sizeof(make_comp_bin_op[0]);

Type random_type() {
    Type T = expr_types[rng()%expr_type_count];
    return T;
}

Expr avg(Expr a, Expr b) {
    Type wider = a.type().with_bits(a.type().bits() * 2);
    return cast(a.type(), (cast(wider, a) + b + 1)/2);
}

Expr random_expr_inner(vector<Expr> inputs, int depth, int func_size);

Expr random_condition(vector<Expr> inputs, int depth, int func_size) {
    Expr a = random_expr_inner(inputs, depth, func_size);
    Expr b = random_expr_inner(inputs, depth, func_size);
    int op = rng() % comp_bin_op_count;
    return make_comp_bin_op[op](a, b);
}

// takes a vector of inputs (points in functions) and an expected Type
// if the chosen input is not of the given type, cast it to conform
Expr make_leaf(vector<Expr> inputs) {
    auto chosen_input = inputs[rand_int(0, inputs.size()-1)];
    return chosen_input;
}

Expr random_expr_inner(vector<Expr> inputs, int depth, int func_size) {
    const int op_count = bin_op_count + bool_bin_op_count + 9;
    const int func_size_thresh = 1e4; // if input is too large do not use trig functions

    if (depth <= 0) {
        return make_leaf(inputs);
    }

    // pick a random operation to combine exprs
    int op = rng() % op_count; // ops need to be defined
    switch(op) {
    case 0:  // casting
    {
        // Get a random type
        Type convertT = random_type();
        auto e1 = random_expr_inner(inputs, depth, func_size);
        return cast(convertT, e1);
    }
    case 1: // select operation
    {
        auto c = random_condition(inputs, depth-2, func_size); // arbitrarily chose to make condition expression shorter
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        auto e2 = random_expr_inner(inputs, depth-2, func_size);
        // make sure e1 and e2 have the same type
        if (e1.type() != e2.type()) {
            e2 = cast(e1.type(), e2);
        }
        return select(c, e1, e2);
    }
    case 2: // unary boolean op
    {
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        if (e1.type().is_bool()) {
            return !e1;
        }
        break;
    }
    case 3: // sin
    {
        if (func_size > func_size_thresh)
            break;
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        return sin(cast<float>(e1));
    }
    case 4: // tanh
    {
        if (func_size > func_size_thresh) {
            // Don't use expensive ops if the function is very large
            break;
        }
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        return tanh(cast<float>(e1));
    }
    case 5: // exp
    {
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        return fast_exp(cast<float>(e1));
    }
    case 6: // sqrt
    {
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        return sqrt(cast<float>(e1));
    }
    case 7: // log
    {
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        return fast_log(cast<float>(e1));
    }
    case 8: // condition
    {
        return random_condition(inputs, depth-1, func_size);
    }
    default: // binary op
        make_bin_op_fn maker;
        auto e1 = random_expr_inner(inputs, depth-1, func_size);
        auto e2 = random_expr_inner(inputs, depth-2, func_size);
        if (e1.type().is_bool() && e2.type().is_bool()) {
            maker = make_bool_bin_op[op % bool_bin_op_count];
        } else {
            maker = make_bin_op[op % bin_op_count];
        }

        return maker(e1, e2);
    }

    // selected case did not return an expression, try again
    return random_expr_inner(inputs, depth, func_size);
}

Expr rand_value(Type t) {
    if (t.is_bool()) {
        return cast(t, rand_int(0,1));
    } else if (t.is_int() || t.is_uint()) {
        return cast(t, rand_int(1, 127));
    } else if (t.is_float()) {
        return cast(t, rand_float());
    } else {
        // Shouldn't get here.
        assert(false);
        return undef(t);
    }
}

Expr random_expr(vector<Expr> inputs, int depth, int func_size) {
    for (auto &e : inputs) {
        e = Internal::simplify(e);
    }

    for (int attempts = 0; attempts < 10; attempts++) {
        Expr result =
            Internal::simplify(Internal::common_subexpression_elimination(random_expr_inner(inputs, depth, func_size)));

        class Checker : public Internal::IRMutator {
        public:
            Expr mutate(const Expr &e) override {
                exprs_to_find.erase(e);
                return IRMutator::mutate(e);
            }
            using Internal::IRMutator::mutate;
            std::set<Expr, Internal::IRDeepCompare> exprs_to_find;
            Checker(const vector<Expr> &inputs) {
                for (const auto &e : inputs) {
                    exprs_to_find.insert(e);
                }
            }
        } checker(inputs);

        checker.mutate(result);

        // Double check all the inputs are used
        if (!checker.exprs_to_find.empty()) {
            std::cerr << "In random expression: " << result << "\n"
                      << "The following expressions were unused:\n";
            for (auto &e : checker.exprs_to_find) {
                std::cerr << e << "\n";
            }
        } else {
            return result;
        }
    }

    // We're having a hard time generating an expression that uses all the inputs. Just sum them.
    Type t = inputs[0].type();
    if (t.is_bool()) {
        t = UInt(8);
    }
    Expr result = cast(t, 0);
    for (const auto &e : inputs) {
        result += e;
    }
    return result;
}

static void hash_combine(uint64_t &h, uint64_t next) {
    // From boost
    h ^= (next + 0x9e3779b9 + (h<<6) + (h>>2));
}

// Generator to produce a random pipeline. The generated pipeline will
// be solely a function of the seed and the number of stages.
// Modified from random_pipeline_generator used by autoscheduler to have 
// learnable parameters (currently just the weights used by the conv stages)
template<bool training>
class RandomPipeline : public Halide::Generator<RandomPipeline<training>> {
public:    
    template<typename T> using Input = GeneratorInput<T>;
    template<typename T> using Output = GeneratorOutput<T>;
    using dim_shape = std::tuple<int,int>;
    using Generator<RandomPipeline<training>>::auto_schedule;
    using Generator<RandomPipeline<training>>::get_pipeline;
    // types for buffers
    using inputT = int16_t;
    Type inputHT = Halide::type_of<inputT>();
    using outputT = int16_t;
    using lossT = float;
    using paramT = float;
    Type paramHT = Halide::type_of<paramT>();

    int num_stage_types = 4;
    const static int num_input_buffers = 4;

    // The random seed to use to generate the pipeline.
    GeneratorParam<int> seed{"seed", 1};
    // The number of input buffers to this->random pipeline
    // The size of the input buffers ASSUMING ALL ARE THE SAME SIZE FOR NOW
    GeneratorParam<int> input_w{"input_w", 14};
    GeneratorParam<int> input_h{"input_h", 14};
    GeneratorParam<int> input_c{"input_c", 3};
    GeneratorParam<int> output_w{"output_w", 10};
    GeneratorParam<int> output_h{"output_h", 10};
    GeneratorParam<int> output_c{"output_c", 3};
    // The number of output buffers to this->random pipeline
    GeneratorParam<int> num_output_buffers{"num_output_buffers", 1};
    // The approximate max number of stages to generate in the random pipeline.
    GeneratorParam<int> max_stages{"max_stages", 20};
    // how much to shift input image by to avoid boundary issues 
    GeneratorParam<int> shift{"shift", 2}; 
    
    Input<int> batch_size{ "batch_size", 1 };
    Input<float> learning_rate{ "learning_rate", 1.0f };
    Input<int> timestep{ "timestep", 0 }; // Needed by ADAM
  
    // store generated pipeline information
    vector<DAGSchema> dag_schema;
    vector<FuncDefSchema> func_def_schema;

    // files for databse information on generated pipelines
    string DAG_csv;
    string FuncDef_csv;

    // for avoiding duplicates 
    std::unordered_map<uint64_t, int>* hashes;

    vector<int> correct_output_type;

    int rejection_count = 0;

    void set_dag_file(string fname) {
        DAG_csv = fname;
    }

    void set_funcdef_file(string fname) {
        FuncDef_csv = fname;
    }

    void set_hashes(std::unordered_map<uint64_t, int>* used_hashes) {
        hashes = used_hashes;
    }

    void do_random_pipeline_schedule(Halide::Pipeline p) {
        // Compute an environment
        std::map<string, Function> env;
        for (Func &f : p.outputs()) {
            std::map<string, Function> more_funcs = find_transitive_calls(f.function());
            env.insert(more_funcs.begin(), more_funcs.end());
        }

        for (auto &f : env) {
            Func(f.second).compute_root();
        }
        return;
    }

    void set_input_weight_shape(Input<Halide::Buffer<float>>* weight, 
                                dim_shape s0, 
                                dim_shape s1, 
                                dim_shape s2, 
                                dim_shape s3) {
        weight->dim(0).set_bounds(std::get<0>(s0), std::get<1>(s0));
        weight->dim(1).set_bounds(std::get<0>(s1), std::get<1>(s1));
        weight->dim(2).set_bounds(std::get<0>(s2), std::get<1>(s2));
        weight->dim(3).set_bounds(std::get<0>(s3), std::get<1>(s3));
    }

    void set_output_weight_shape(Output<Halide::Buffer<paramT>>* weight,
                                 dim_shape s0,
                                 dim_shape s1,
                                 dim_shape s2,
                                 dim_shape s3) {
        weight->dim(0).set_bounds(std::get<0>(s0), std::get<1>(s0));
        weight->dim(0).set_bounds_estimate(std::get<0>(s0), std::get<1>(s0));
        weight->bound(weight->args()[0], std::get<0>(s0), std::get<1>(s0));
        weight->estimate(weight->args()[0], std::get<0>(s0), std::get<1>(s0));

        weight->dim(1).set_bounds(std::get<0>(s1), std::get<1>(s1));
        weight->dim(1).set_bounds_estimate(std::get<0>(s1), std::get<1>(s1));
        weight->bound(weight->args()[1], std::get<0>(s1), std::get<1>(s1));
        weight->estimate(weight->args()[1], std::get<0>(s1), std::get<1>(s1));

        weight->dim(2).set_bounds(std::get<0>(s2), std::get<1>(s2));
        weight->dim(2).set_bounds_estimate(std::get<0>(s2), std::get<1>(s2));
        weight->bound(weight->args()[2], std::get<0>(s2), std::get<1>(s2));
        weight->estimate(weight->args()[2], std::get<0>(s2), std::get<1>(s2));
        
        weight->dim(3).set_bounds(std::get<0>(s3), std::get<1>(s3));
        weight->dim(3).set_bounds_estimate(std::get<0>(s3), std::get<1>(s3));
        weight->bound(weight->args()[3], std::get<0>(s3), std::get<1>(s3));
        weight->estimate(weight->args()[3], std::get<0>(s3), std::get<1>(s3));

        weight->dim(weight->dimensions()-1).set_bounds(0, 4);
        weight->dim(weight->dimensions()-1).set_bounds_estimate(0, 4);
    }

    void backprop(Halide::ImageParam &weights,
                  Output<Halide::Buffer<paramT>>* grad, 
                  const Derivative &d, 
                  Expr learning_rate, 
                  Expr timestep) {
        std::vector<Expr> args(weights.dimensions()+1);
        for (auto &e : args) e = Var();
        (*grad)(args) = undef<paramT>();
        // We'll report back the new weights and the loss gradients,
        // and update the ADAM state. Depending on the mode the caller
        // is in, it may use the new weights, or it may just send the
        // loss gradients up to an ADAM server.
        args.back() = 0;
        FuncRef new_weight = (*grad)(args);
        args.back() = 1;
        FuncRef smoothed_deriv = (*grad)(args);
        args.back() = 2;
        FuncRef smoothed_second_moment = (*grad)(args);
        args.back() = 3;
        FuncRef loss_gradient = (*grad)(args);

        args.pop_back();
        Expr current_weight = weights(args);

        loss_gradient = d(weights)(args);
        std::cout << "loss gradient: " << loss_gradient << std::endl;
        std::cout << "loss gradient update definitons: " << std::endl;
        for (auto& def : loss_gradient.function().updates()) {
            for (auto& expr : def.values()) {
                std::cout << expr << std::endl;
            }
        }

        // Update the first and second moment estimates
        //smoothed_deriv = 0.9f * smoothed_deriv + 0.1f * loss_gradient;
        std::cout << "\nsmoothed deriv: " << smoothed_deriv << std::endl;
        std::cout << "smoothed deriv update definitons: " << std::endl;
        for (auto& def : smoothed_deriv.function().updates()) {
            for (auto& expr : def.values()) {
                std::cout << expr << std::endl;
            }
        }
        //smoothed_second_moment = 0.999f * smoothed_second_moment + 0.001f * pow(loss_gradient, 2);
      
        // Correction to account for the fact that the smoothed_deriv
        // and smoothed_second_moment start at zero when t == 0
        Expr smoothed_deriv_correction = 1 / (1 - pow(0.9f, timestep + 1));
        Expr smoothed_second_moment_correction = 1 / (1 - pow(0.999f, timestep + 1));

        std::cout << "\nsmoothed deriv expr: " << (Expr)smoothed_deriv << std::endl;
        std::cout << smoothed_deriv.function().name() << std::endl;
        // Update the weights
        //Expr step = learning_rate * smoothed_deriv * smoothed_deriv_correction;
        //step /= sqrt(smoothed_second_moment * smoothed_second_moment_correction) + 1e-5f;
        Expr step = learning_rate * d(weights)(args);

        std::cout << "step: " << step << std::endl;
        new_weight = current_weight - step;
    }

    void set_upcast_types(Type input_type, Type& mult_type, Type& sum_type) {
        if (input_type.is_bool()) {
            mult_type = UInt(8);
            sum_type = UInt(8);
        } else if (!input_type.is_float() && rand_int(0,1)) {
            int input_bits = input_type.bits();
            int mult_bits = std::min(32, 2*input_bits);
            int sum_bits = std::min(32, 2*mult_bits);
            mult_type = input_type.with_bits(mult_bits);
            sum_type = input_type.with_bits(sum_bits);
        } else {
            mult_type = input_type;
            sum_type = input_type;
        }
        return;
    }

    void set_downcast_type(Type input_type, Type& output_type) {
        if (input_type.is_int() && rand_int(0,1)) {
            int input_bits = input_type.bits();
            int factor = rand_int(1, 2) * 2;
            int output_bits = std::max(8, input_bits/factor);
            output_type = Int(output_bits);
        } else {
            output_type = input_type;
        }
        return;
    }

    static std::string type_string(vector<int>& type) {
        std::string type_string = "";
        for (int v : type) {
            type_string += std::to_string(v);
        }
        return type_string;
    }

    static bool type_match(vector<int>& typeA, vector<int>& typeB) {
        bool match = true;
        for (int c = 0; c < 3; c++) {
            match = match && ( typeA[c] == typeB[c] );
        }
        return match;
    }

    class Stage {
        public:
            uint64_t stage_index;
            uint64_t stage_type;
            uint64_t hash;
            RandomPipeline<training>* gen;
            
            vector<int> output_type;

            virtual vector<int>& compute_output_type() {
                assert(!output_type.empty());
                return output_type;
            };

            void add_dag_schema(Stage producer) {
                vector<int>& type = compute_output_type();
                gen->dag_schema.emplace_back((uint64_t)gen->seed, std::string(func.name()), 
                                        stage_type, stage_index, type_string(type), 
                                        producer.stage_index,
                                        std::string(producer.func.name()));
            }
            
            void add_func_def_schema(Expr &value, vector<Var> args) {
                std::ostringstream left;
                left << func(args);
                const auto left_string = left.str();

                std::ostringstream right;
                right << value;
                const auto right_string = right.str();

                gen->func_def_schema.emplace_back((uint64_t)gen->seed, std::string(func.name()), 
                                             stage_index, left_string + " = " + right_string); 
            }
            
            Func func;
            // approx width and height and channels. Used to preserve spatial
            // scale when combining stages, and to track the total sizes of things.
            int w, h, c;

            static constexpr int max_size = 10000;
            static constexpr int min_size = 100;
            static constexpr int max_stride = 3; // for convs and pools

            Stage() {}
            Stage(Func f, int w, int h, int c, vector<int> output_type,
                  RandomPipeline<training>* gen)  : 
                func(f), w(w), h(h), c(c), output_type(output_type), gen(gen) {
                stage_type = 0;
                stage_index = 0;
            }

            int size() const {
                return w*h*c;
            }

            bool may_increase_size() const {
                return size() < max_size && w <= 8000 && h <= 8000 && c <= 512;
            }

            bool may_reduce_size() const {
                return size() > min_size;
            }

            int random_size_increase_factor() const {
                int sz = size();
                int max_factor = (max_size + sz - 1) / sz;
                if (max_factor <= 1) return 1;
                int log_max_factor = std::ceil(std::log(max_factor) / std::log(2));
                int factor = 1 << rand_int(std::max(1, log_max_factor - 3), log_max_factor);
                return factor;
            }

            int random_size_reduce_factor() const {
                int sz = size();
                int max_factor = (sz + min_size - 1) / min_size;
                if (max_factor <= 1) return 1;
                return std::min(8, 1 << rand_int(1, std::ceil(std::log(max_factor) / std::log(2))));
            }

            int random_out_channels() const {
                int min = (min_size + w * h - 1) / (w * h);
                int max = std::min(512, max_size / (w * h));
                if (min >= max) return min;
                return rand_int(min, max);
            }
    };

    vector<Stage> stages; // the list of stages we will generate

    // USED BY INTERP 2TAP STAGES
    typedef std::tuple<Stage, vector<Expr>, vector<Expr>, Func> InterpStageAndCoords;

    /** generates interpolation coords and makes sure that the coordinates are not the same **/
    static bool random_coords(vector<Expr> &coords1, vector<Expr> &coords2, uint64_t &h1, uint64_t &h2) {
        int choice = rand_int(0, 2);
        int offset11, offset12, offset21, offset22;
        offset11 = offset12 = offset21 = offset22 = 1;

        switch (choice) {
            case 0:
                break; 
            case 1:
                offset11 = 2;
                coords1[0] += 1;
                break;
            case 2:
                offset11 = 0;
                coords1[0] -= 1;
        }
        choice = rand_int(0, 2);
        switch (choice) {
            case 0:
                break; 
            case 1:
                offset12 = 2;
                coords1[1] += 1;
                break;
            case 2:
                offset12 = 0;
                coords1[1] -= 1;
        }
        choice = rand_int(0, 2);
        switch (choice) {
            case 0:
                break; 
            case 1:
                offset21 = 2;
                coords2[0] += 1;
                break;
            case 2:
                offset21 = 0;
                coords2[0] -= 1;
        }
        choice = rand_int(0, 2);
        switch (choice) {
            case 0:
                break; 
            case 1:
                offset22 = 2;
                coords2[1] += 1;
                break;
            case 2:
                offset22 = 0;
                coords2[1] -= 1;
        }
        
        hash_combine(h1, (offset11)*10 + (offset12));
        hash_combine(h2, (offset21)*10 + (offset22));

        bool success = !( equal(coords1[0], coords2[0]) && equal(coords1[1], coords2[1]) );
        return success;
    }

    class Interp2Tap : public Stage {
        public:
        Stage input_stage;
        vector<Expr> input_coords1;
        vector<Expr> input_coords2;
        
        vector<int>& compute_output_type() {
            if (this->output_type.empty()) {
                this->output_type = input_stage.compute_output_type();
            }
            return this->output_type;
        }
        
        Interp2Tap() {};

        Interp2Tap(vector<Stage> &s, uint64_t h,  
                   RandomPipeline<training>* gen,
                   int input_id = -1) {
            this->hash = h;
            this->gen = gen;
            this->stage_index = (uint64_t)s.size() - num_input_buffers + 1;
            this->stage_type  = (uint64_t)1;
            Func interp("interp2Tap");
            if (input_id < 0) {
                // pick a random input
                input_id = rand_int(0, s.size()-1);
            }
            input_stage = s[input_id];
            Func input_func = input_stage.func;
            std::cout << interp.name() << " is Interp 2 tap on " << input_func.name() << std::endl;

            // generate random coordinates to use 
            input_coords1 = make_arguments(input_func.args());
            input_coords2 = make_arguments(input_func.args());
            uint64_t h_coords1, h_coords2;
            h_coords1 = h_coords2 = 0;
            while (!random_coords(input_coords1, input_coords2, h_coords1, h_coords2)) {
                input_coords1 = make_arguments(input_func.args());
                input_coords2 = make_arguments(input_func.args());
                h_coords1 = h_coords2 = 0;
            }

            Expr value = avg(input_func(input_coords1), input_func(input_coords2));
            interp(input_func.args()) = value;

            std::cout << interp(input_func.args()) << " = ";
            std::cout << value << std::endl;

            this->func = interp;
            this->w = input_stage.w;
            this->h = input_stage.h;
            this->c = input_stage.c;
    
            hash_combine(this->hash, this->stage_type); 
            hash_combine(this->hash, input_id);
            hash_combine(this->hash, h_coords1 + h_coords2);
            
            this->add_dag_schema(input_stage);
            this->add_func_def_schema(value, input_func.args());
        }            
        // end of public methods
    };

    static bool same_vars(vector<Var> v1, vector<Var> v2) {
        assert(v1.size() == v2.size());
        for (int i = 0; i < (int)v1.size(); i++) {
            if (v1[i].name() != v2[i].name()) return false;
        }
        return true;
    }

    class SelectInterp2Tap : public Stage {
        public:
        Interp2Tap interp1_stage, interp2_stage;
        
        vector<int>& compute_output_type() {
            if (this->output_type.empty()) {
                vector<int>& interp1_output_type = interp1_stage.compute_output_type();
                vector<int>& interp2_output_type = interp2_stage.compute_output_type();
                if (interp1_output_type != interp2_output_type) {
                    throw std::runtime_error( "select must choose from interps of same type" );
                } 
                this->output_type = interp1_output_type;
            }
            return this->output_type;
        }

        SelectInterp2Tap(vector<Stage> &s, uint64_t h,
                         RandomPipeline<training>* gen,
                         int input_id = -1) {
            this->hash = h;
            this->gen = gen;
            this->stage_type  = (uint64_t)2;
            Func selectInterp("selectInterp2Tap");

            uint64_t h_interp1, h_interp2;
            h_interp1 = h_interp2 = 0;

            std::cout << selectInterp.name() << " is Select Interp" << std::endl; 

            Func interp1_input, interp2_input;
            vector<Expr> interp1_coords1, interp1_coords2, interp2_coords1, interp2_coords2;

            interp1_stage = Interp2Tap(s, h_interp1, this->gen, input_id);
            interp1_input = interp1_stage.input_stage.func;
            interp1_coords1 = interp1_stage.input_coords1;
            interp1_coords2 = interp1_stage.input_coords2;
            s.push_back(interp1_stage);

            interp2_stage = Interp2Tap(s, h_interp2, this->gen);
            interp2_input = interp2_stage.input_stage.func;
            interp2_coords1 = interp2_stage.input_coords1;
            interp2_coords2 = interp2_stage.input_coords2;
            s.push_back(interp2_stage);
          
            this->stage_index = (uint64_t)s.size() - num_input_buffers + 1;

            std::cout << selectInterp.name() << " selects from: " << interp1_stage.func.name() << " and " << interp2_stage.func.name() << std::endl; 

            Expr diff1 = absd(interp1_input(interp1_coords1), interp1_input(interp1_coords2));      
            Expr diff2 = absd(interp2_input(interp2_coords1), interp2_input(interp2_coords2));

            vector<Var> args = interp1_stage.func.args(); 

            Expr value = select(diff1 < diff2, interp1_stage.func(args), interp2_stage.func(args));
            selectInterp(args) = value;

            std::cout << selectInterp(args) << " = ";
            std::cout << value << std::endl;

            this->func = selectInterp;
            this->w = interp1_stage.w;
            this->h = interp1_stage.h;
            this->c = interp1_stage.c;
            
            hash_combine(this->hash, this->stage_type); 
            hash_combine(this->hash, interp1_stage.hash + interp2_stage.hash);
            this->add_dag_schema(interp1_stage);
            this->add_dag_schema(interp2_stage);
            this->add_func_def_schema(value, args);
        }
    };

    class CorrectInterp2Tap : public Stage {
        public:
        Stage ref_stage, interp_stage, input_stage;
        vector<Expr> coords1; // coordinates used to index
        vector<Expr> coords2; // interp stage and input stage

        vector<int>& compute_output_type() {
            if (this->output_type.empty()) {
                vector<int>& ref_output_type = ref_stage.compute_output_type();
                vector<int>& interp_output_type = interp_stage.compute_output_type();
                vector<int>& input_output_type = input_stage.compute_output_type();
                
                assert(ref_output_type.size() == interp_output_type.size() && 
                       ref_output_type.size() == input_output_type.size());


                this->output_type = ref_output_type;

                for (int i = 0; i < ref_output_type.size(); i++) {
                    this->output_type[i] -= interp_output_type[i];
                    this->output_type[i] += input_output_type[i];
                }
            }
            return this->output_type;
        }
      
        CorrectInterp2Tap() {};
 
        CorrectInterp2Tap(vector<Stage> &s, uint64_t h, 
                          RandomPipeline<training>* gen,
                          int use_id=-1) {
            this->hash = h;
            this->gen = gen;
            this->stage_index = (uint64_t)s.size() - num_input_buffers + 1;
            this->stage_type = 3;
            Func correctInterp("correctInterp2Tap");

            int input_id, ref_id, interp_id;

            // pick a random input buffers
            input_id = rand_int(0, s.size() - 1);
            ref_id = rand_int(0, s.size() - 1);
            interp_id = rand_int(0, s.size() - 1);

            // if stage id is given, use that as one of the input functions 
            if (use_id >= 0) {
                // pick a buffer to fill given input
                int buff_id = rand_int(0, 2);
                switch (buff_id) {
                    case 0:
                        input_id = use_id;
                        break;
                    case 1: 
                        ref_id = use_id;
                        break;
                    case 2:
                        interp_id = use_id;
                        break;
                }
            }

            input_stage = s[input_id];
            ref_stage = s[ref_id];
            interp_stage = s[interp_id];

            Func input_f  = input_stage.func;
            Func ref_f    = ref_stage.func;
            Func interp_f = interp_stage.func;
            
            std::cout << correctInterp.name() << " is Corrected Interp 2 Tap on: " << input_f.name() << " with correction funcs: " << ref_f.name() << " and " << interp_f.name() << std::endl;

            // generate random coordinates to use 
            coords1 = make_arguments(input_f.args());
            coords2 = make_arguments(input_f.args());

            uint64_t h_coords1, h_coords2;
            h_coords1 = h_coords2 = 0;
            while (!random_coords(coords1, coords2, h_coords1, h_coords2)) {
                coords1 = make_arguments(input_f.args());
                coords2 = make_arguments(input_f.args());
                h_coords1 = h_coords2 = 0;
            }

            vector<Expr> coords = make_arguments(input_f.args()); 
            Expr correction = ref_f(coords) - avg(interp_f(coords1), interp_f(coords2));
            Expr value = correction + avg(input_f(coords1), input_f(coords2));

            correctInterp(coords) = value;
            
            std::cout << correctInterp(coords) << " = ";
            std::cout << value << std::endl;

            this->func = correctInterp;
            this->w = input_stage.w;
            this->h = input_stage.h;
            this->c = input_stage.c;

            hash_combine(this->hash, this->stage_type);
            hash_combine(this->hash, input_id);
            hash_combine(this->hash, ref_id);
            hash_combine(this->hash, interp_id);
            hash_combine(this->hash, h_coords1 + h_coords2);

            this->add_dag_schema(input_stage);
            this->add_dag_schema(ref_stage);
            this->add_dag_schema(interp_stage);
            this->add_func_def_schema(value, input_f.args());
        } 
    };

    class SelectCorrectInterp : public Stage {
        public:
        CorrectInterp2Tap correctInterp1, correctInterp2;
        
        vector<int>& compute_output_type() {
            if (this->output_type.empty()) {
                vector<int>& correctInterp1_type = correctInterp1.compute_output_type();
                vector<int>& correctInterp2_type = correctInterp2.compute_output_type();
                if (correctInterp1_type != correctInterp2_type) {
                    throw std::runtime_error( "select must choose from interps of same type" );
                } 
                this->output_type = correctInterp1_type;
            }
            return this->output_type;
        }
        
        SelectCorrectInterp(vector<Stage> &s, uint64_t h,
                            RandomPipeline<training>* gen,
                            int input_id=-1) {
            this->hash = h;
            this->gen = gen;
            this->stage_type = 4;
            Func selectCorrectInterp("selectCorrectInterp2Tap");
            std::cout << selectCorrectInterp.name() << " is Select Corrected Interp" << std::endl;

            vector<Expr> s1coords1, s1coords2, s2coords1, s2coords2;
            Func s1input, s2input;

            uint64_t h_interp1, h_interp2;
            h_interp1 = h_interp2 = 0;

            correctInterp1 = CorrectInterp2Tap(s, h_interp1, this->gen, input_id);
            s.push_back(correctInterp1);
            correctInterp2 = CorrectInterp2Tap(s, h_interp2, this->gen);
            s.push_back(correctInterp2);
          
            std::cout << selectCorrectInterp.name() << " selects from: " << correctInterp1.func.name() << " and " << correctInterp2.func.name() << std::endl; 
            this->stage_index = (uint64_t)s.size() - num_input_buffers + 1;

            Func correctInterp1_input = correctInterp1.input_stage.func;
            Func correctInterp2_input = correctInterp2.input_stage.func;

            assert(same_vars(correctInterp1.func.args(), correctInterp2.func.args()));
            assert(correctInterp1.w == correctInterp2.w 
                && correctInterp1.h == correctInterp2.h 
                && correctInterp1.c == correctInterp2.c);

            Expr diff1 = absd(correctInterp1_input(correctInterp1.coords1), correctInterp1_input(correctInterp1.coords2));
            Expr diff2 = absd(correctInterp2_input(correctInterp2.coords1), correctInterp2_input(correctInterp2.coords2));

            vector<Var> args = correctInterp1.func.args();
            Expr value = select(diff1 < diff2, correctInterp1.func(args), correctInterp2.func(args));
            selectCorrectInterp(args) = value;
            
            std::cout << selectCorrectInterp(args) << " = ";
            std::cout << value << std::endl;
        
            this->func = selectCorrectInterp;
            this->w = correctInterp1.w;
            this->h = correctInterp1.h;
            this->c = correctInterp1.c;

            hash_combine(this->hash, this->stage_type);
            hash_combine(this->hash, h_interp1 + h_interp2);

            this->add_dag_schema(correctInterp1);
            this->add_dag_schema(correctInterp2);
            this->add_func_def_schema(value, args);
        }
    };

    // Add a random new stage onto the end of the pipeline that can choose any of the 
    // input buffers or previous stages as an input. Note that the type of random stage
    // will determine how many inputs it needs 
    Stage random_stage(vector<Stage> &s, uint64_t h, int input_id=-1) {
        int stage_type = rand_int(0, 3); 
        std::cout <<  "STAGE TYPE: " << stage_type << std::endl;
        std::cout.flush();
        if (stage_type == 0) {
            return Interp2Tap(s, h, this, input_id);
        } else if (stage_type == 1) {
            if (s.size() < 2) {
                return random_stage(s, h, input_id);
            }
            return SelectInterp2Tap(s, h, this, input_id);
        } else if (stage_type == 2) {
            if (s.size() < 3) { 
                return random_stage(s, h, input_id);
            }
            return CorrectInterp2Tap(s, h, this, input_id);
        } else if (stage_type == 3) {
            if (s.size() < 3) {
                return random_stage(s, h, input_id);
            }
            return SelectCorrectInterp(s, h, this, input_id);
        }
    }

    void reset() {
        rejection_count++;
        stages.erase(stages.begin()+num_input_buffers, stages.end());
        dag_schema.erase(dag_schema.begin(), dag_schema.end());
        func_def_schema.erase(func_def_schema.begin(), func_def_schema.end());
    }

    // build pipeline and define all required inputs and outputs for the generated program
    void configure() {
        // create input and output buffers
        for (int i = 0; i < num_input_buffers; i++) {
            Input<Buffer<inputT>>* input_buff = 
                Halide::Internal::GeneratorBase::add_input<Buffer<outputT>>("input_" + std::to_string(i), 3);
            input_buffs.push_back(input_buff);
        }
        for (int i = 0; i < num_output_buffers; i++) {
            Output<Buffer<outputT>>* output_buff = 
                Halide::Internal::GeneratorBase::add_output<Buffer<outputT>>("output_" + std::to_string(i), 3);
            output_buffs.push_back(output_buff);
            Input<Buffer<outputT>>* correct_output_buff = 
                Halide::Internal::GeneratorBase::add_input<Buffer<outputT>>("correct_output_" + std::to_string(i), 3);
            correct_outputs.push_back(correct_output_buff);
        }

        rng.seed((int)seed);

        Var x("x"), y("y"), c("c");

        // create dummy image params for each input buffer so that we can access them in configure()
        // Zero pad all inputs and add them as stages to be used by the generated random stages
        // Assuming all inputs are same size for now
        for (int i = 0; i < num_input_buffers; i++) { 
            input_buff_dummies.emplace_back(Halide::ImageParam(inputHT, 3, "input_" + std::to_string(i)));
            std::vector<std::pair<Expr, Expr>> bounds(3); 
            bounds.at(0).first = 0;
            bounds.at(0).second = input_w;
            bounds.at(1).first = 0;
            bounds.at(1).second = input_h;
            bounds.at(2).first = 0;
            bounds.at(2).second = input_c;
            Func padded_input = Halide::BoundaryConditions::constant_exterior(input_buff_dummies[i], cast(inputHT, 0), bounds);
            vector<int> input_type;
            std::string func_name;
            switch (i) {
                case 0:
                    func_name = "shifted_GR";
                    input_type = {0, 1, 0, 0, 0, 0};
                    break;
                case 1:
                    func_name = "shifted_R";
                    input_type = {1, 0, 0, 0, 0, 0};
                    break;
                case 2:
                    func_name = "shifted_B";
                    input_type = {0, 0, 1, 0, 0, 0};
                    break;
                case 3:
                    func_name = "shifted_GB";
                    input_type = {0, 1, 0, 0, 0, 0};
                    break;
            }
            Func shifted_input(func_name);
            // shift the input so that we don't have to worry about boundary conditions
            Expr value = padded_input(x + (int)shift, y + (int)shift, c);
            shifted_input(x, y, c) = value;

            std::cout << shifted_input(x, y, c) << " = " << value << std::endl;
      
            stages.emplace_back(Stage(shifted_input, output_w, output_h, output_c, input_type, this));  
        } 

        std::cout << "max stages: " << (int)max_stages << "\n" << std::endl;
        // NOTE: We cannot stop generating stages until we've created at least enough stages to fill the outputs 
        // for now just randomly assigning generated funcs to outputs but in the future we will need to make 
        // sure that the funcs satisfy the size/type/other constraints on the output buffers. 
        // CONSIDER growing pipeline from output and input buffers.
        assert((int)max_stages >= (int)num_output_buffers);

        // keep generating pipelines until we don't get a duplicate
        while (true) {
            uint64_t h = 0;
            bool type_error = false;
            for (int i = 0; i < max_stages; i++) {
                Stage next;
                if (i > 0) {
                    try {
                        next = random_stage(stages, h, stages.size()-1); // use most recently created func as input
                    } catch (const std::exception& ex) {
                        std::cout << ex.what() << "\npipeline type error, resetting generator..." << std::endl;
                        reset(); 
                        type_error = true;
                        break;
                    }
                } else {
                    try {
                        next = random_stage(stages, h); // use most recently created func as input
                    } catch (const std::exception& ex) {
                        std::cout << ex.what() << "\npipeline type error, resetting generator..." << std::endl;
                        reset(); 
                        type_error = true;
                        break;
                    }
                }
                stages.push_back(next);
                h = stages.back().hash;
                std::cout << "Approx size: " << stages.back().w << ", " << stages.back().h << ", " << stages.back().c << "\n\n";
            }

            if (type_error) continue;

            std::cout << "finished adding stages" << std::endl;

            // check that pipeline is not a duplicate and type check
            vector<int>& output_type = stages.back().compute_output_type();
           
            if (!(*hashes)[h]++ && type_match(output_type, correct_output_type)) {
                break;
            } // else keep generating pipelines
            reset();
        }
    }

    // Select which funcs to map to the output buffers 
    // Compute the loss and call backprop if we are in training mode
    void generate() {
        Var x("x"), y("y"), c("c");

        std::vector<Func> last_funcs; // need these for backrop if training
        last_funcs.push_back(stages[stages.size()-1].func);

        (*output_buffs[0])(x, y, c) = stages[stages.size()-1].func(x, y, c);
      
        Derivative d_loss_d;
        Func err;

        // need to compute total loss over all outputs
        RDom r(0, output_w, 
               0, output_h,
               0, output_c);
        Expr loss = Expr(0.0f);
        for (int i = 0; i < num_output_buffers; i++) {
            Expr diff = cast<double>((*correct_outputs[i])(x, y, c) - last_funcs[i](x, y, c));
            err(x, y, c) = (diff*diff);
            loss += sum(err(r.x, r.y, r.z)/((int)output_w * (int)output_h));
        }

        loss_output() = cast<lossT>(loss);

        // dump the schema information
        std::ofstream dag_file;
        dag_file.open(DAG_csv, std::ofstream::out | std::ofstream::app);

        for ( auto& elem: dag_schema) {
            dag_file << elem.dump() << "\n";
        } 
        dag_file.close(); 

        std::ofstream func_def_file;
        func_def_file.open(FuncDef_csv, std::ofstream::out | std::ofstream::app);

        for ( auto& elem: func_def_schema) {
            func_def_file << elem.dump() << "\n";
        } 
        func_def_file.close(); 

        // Compute derivatives of the loss, and backprop them to the parameters.
        if (training) {
            d_loss_d = propagate_adjoints(loss_output);
            
            // iterate over the generated params and backprop
            for (auto &output_w : output_params) {
                auto& input_w = input_param_dummies[output_w.first];
                backprop(input_w, output_w.second, d_loss_d, learning_rate, timestep);
            }
        }
        // set param_shapes for input and output weights
        if (training) {
            for (auto &output_w : output_params) {
                auto &shape = param_shapes[output_w.first];
                auto input_w = input_params[output_w.first];
                set_input_weight_shape(input_w, std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), std::get<3>(shape));
                set_output_weight_shape(output_w.second, std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), std::get<3>(shape));
            }      
        } else {
            for (auto &input_w : input_params) {
                auto &shape = param_shapes[input_w.first];
                set_input_weight_shape(input_w.second, std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), std::get<3>(shape));
            }      
        }
        learning_rate.set_estimate(0.001f);
        timestep.set_estimate(37);
        batch_size.set_estimate(1);

        // SCHEDULING
        if (!auto_schedule and !training) {
            do_random_pipeline_schedule(get_pipeline());
        } 
        if (!auto_schedule and training) {
            do_random_pipeline_schedule(get_pipeline());
        }

        // bound all inputs and outputs
        for (int i = 0; i < num_input_buffers; i++) {
            (*input_buffs[i]).dim(0).set_bounds_estimate(0, input_w)
                .dim(1).set_bounds_estimate(0, input_h)
                .dim(2).set_bounds_estimate(0, input_c);
        }
        for (int i = 0; i < num_output_buffers; i++) {
            (*correct_outputs[i]).dim(0).set_bounds_estimate(0, output_w)
                .dim(1).set_bounds_estimate(0, output_h)
                .dim(2).set_bounds_estimate(0, output_c);

            (*output_buffs[i]).dim(0).set_bounds_estimate(0, output_w)
                .dim(1).set_bounds_estimate(0, output_h)
                .dim(2).set_bounds_estimate(0, output_c);
        }
    }

    void set_inputs(const std::vector<Buffer<inputT>> &inputs) {
        for (size_t i = 0; i < inputs.size(); i++) input_buff_dummies[i].set(inputs[i]);
    }
    
    void set_correct_output_type(vector<int> &type) {
        correct_output_type = type;
        std::cout <<  "setting output type " << std::endl;
        for (auto v: correct_output_type) std::cout << v;
    }

private:
    std::vector<Halide::ImageParam> input_buff_dummies;
    std::vector<Input<Buffer<inputT>> *>   input_buffs;
    std::vector<Input<Buffer<outputT>> *>  correct_outputs;
    std::vector<Output<Buffer<outputT>> *> output_buffs;

    std::unordered_map<string, Halide::ImageParam> input_param_dummies;
    std::unordered_map<string, Input<Halide::Buffer<paramT>> *> input_params;
    std::unordered_map<string, Output<Halide::Buffer<paramT>> *> output_params;
    // param_shapes of parameter buffers
    std::unordered_map<string, std::tuple<dim_shape, dim_shape, dim_shape, dim_shape>> param_shapes;

    Output<Buffer<lossT>> loss_output { "loss_output", 0 };
};

using RandomPipelineInference = RandomPipeline<false>;
using RandomPipelineTraining = RandomPipeline<true>;


HALIDE_REGISTER_GENERATOR(RandomPipelineInference, random_pipeline_inference)
HALIDE_REGISTER_GENERATOR(RandomPipelineTraining, random_pipeline_training)
