#ifndef NEURAL_NETWORK_2_H_INCLUDED
#define NEURAL_NETWORK_2_H_INCLUDED

#include <utility> // for std::pair

#include "../../libmda/util/random_bool.h"
#include "../../libmda/util/stacktrace.h"
#include "../../libmda/numeric/optim/wolfe_line_search.h"
#include "training_set.h"
#include "neuron.h"
#include "pgm_image_train.h"
#include "neuron_function.h"

using weight_idx_pair_t = std::pair<int,int>;
using feed_t = std::vector<std::vector<double> >;

/**
 *
 **/
std::ostream& operator<<(std::ostream& os, const weight_idx_pair_t& p)
{
   os << "{" << p.first << "," << p.second << "}";
   return os;
}

/**
 *
 **/
class neural_network_evaluator
{
   private:
      std::vector<int> m_n_layer_units;
      std::vector<std::unique_ptr<neuron_function> > m_layer_neuron_function;
      std::vector<std::vector<weight_idx_pair_t> > m_weight_indices;
      double m_gamma = 1e-8; // weight decay coef
      //double m_gamma = 0.0;
      
      /**
       *
       **/
      void assert_training_set( const training_set_t& t ) const
      {
         for(int i = 0; i < t.size(); ++i)
         {
            assert(t[i].first.size() == m_n_layer_units[0]);
            assert(t[i].second.size() == m_n_layer_units[m_n_layer_units.size() - 1]);
         }
      }

      /**
       *
       **/
      void assert_layer_input( const weight_idx_pair_t& idx, const std::vector<double>& layer_input ) const
      {
         int size = idx.second - idx.first - 1;
         //assert(layer_input.size() == size);
         if(layer_input.size() != size)
         {
            libmda::util::print_stacktrace();
            exit(40);
         }
      }
      
   public:
      /**
       * constructor
       * @n_layer_units number of units in each layer, and implicitly defines number of layers
       * @layer_functions defines squashing function for each layer
       **/
      neural_network_evaluator
         ( const std::vector<int>& n_layer_units
         , const std::vector<std::string>& layer_functions
         )
         : m_n_layer_units(n_layer_units)
         , m_layer_neuron_function(n_layer_units.size())
      {
         assert(m_n_layer_units.size() > 1); // we must have at least 2 layers (input and output)
         assert(layer_functions.size() > 0); // we must have at least 1 layer function (output)
         assert(m_n_layer_units.size() == (layer_functions.size() + 1)); // we must have layer functions for each hidden layer + output
         
         // init weight_indices
         int weight_begin = 0;
         m_weight_indices.resize(m_n_layer_units.size());
         for(int i_layer = 1; i_layer < m_n_layer_units.size(); ++i_layer)
         {
            int n_weights = m_n_layer_units[i_layer - 1] + 1;
            
            for(int i_unit = 0; i_unit < m_n_layer_units[i_layer]; ++i_unit)
            {
               int weight_end = weight_begin + n_weights;
               m_weight_indices[i_layer].emplace_back(weight_begin, weight_end);
               weight_begin = weight_end;
            }
         }

         // setup layer unit functions 
         // (each layer has only type of function, could be expanded such that each unit can have a unique function)
         // (then access with [i_layer][i_unit])
         m_layer_neuron_function[0] = neuron_function::factory("null"); // input layer has no squashing function
         for(int i_layer = 1; i_layer < m_layer_neuron_function.size(); ++i_layer)
         {
            m_layer_neuron_function[i_layer] = neuron_function::factory(layer_functions[i_layer - 1]);
         }
      }

      /**
       *
       **/
      int num_layers() const
      {
         return m_n_layer_units.size();
      }

      /**
       *
       **/
      int num_weights() const
      {
         // calculate number of weights
         int n_weights = 0;
         for(int i = 1; i < m_n_layer_units.size(); ++i)
         {
            n_weights += m_n_layer_units[i]*(m_n_layer_units[i - 1] + 1);
         }
         return n_weights;
      }

      /**
       *
       **/
      int largest_layer() const
      {
         int max = 0;
         for(int i = 0; i < m_n_layer_units.size(); ++i)
         {
            max = std::max(max,m_n_layer_units[i]);
         }
         return max;
      }
      
      /**
       * @i_weight the weight to get
       * @i_unit the unit
       * @i_layer the layer
       * @weights from where to get the weight
       **/
      double weight( int i_layer
                   , int i_unit
                   , int i_weight
                   , const std::vector<double>& weights
                   ) const
      {
         auto& weight_idx_pair = m_weight_indices[i_layer][i_unit];
         return weights[weight_idx_pair.first + i_weight + 1]; // +1 for bias
      }
      
      /**
       * unit dot
       **/
      double unit_dot( int layer
                     , int unit
                     , const std::vector<double>& weights
                     , const std::vector<double>& layer_input 
                     ) const
      {
         auto& weight_idx_pair = m_weight_indices[layer][unit];
         
         assert_layer_input(weight_idx_pair, layer_input);
         
         double unit_dot = 0.0;
         unit_dot += weights[weight_idx_pair.first]; // add bias
         int layer_input_idx = 0;
         for(int i = weight_idx_pair.first + 1; i < weight_idx_pair.second; ++i) // calculate dot(x,w)
         {
            unit_dot += layer_input[layer_input_idx]*weights[i];
            ++layer_input_idx;
         }
         assert(layer_input_idx == layer_input.size()); // assert correct sizes
         return unit_dot;
      }

      /**
       * 
       **/
      double unit_evaluate( int i_layer
                          , int i_unit
                          , const std::vector<double>& weights
                          , const std::vector<double>& layer_input
                          ) const
      {
         double unit_dot_result = unit_dot(i_layer, i_unit, weights, layer_input);
         //return 1.0 / (1.0 + std::exp(-unit_dot_result)); // HARDCODED: sigmoid unit
         return m_layer_neuron_function[i_layer]->evaluate(unit_dot_result);
      }
      
      /**
       *
       **/
      double unit_derivative( int i_layer
                            , int i_unit
                            , const std::vector<double>& weights
                            , const std::vector<double>& layer_input
                            ) const
      {
         double unit_dot_result = unit_dot(i_layer, i_unit, weights, layer_input);
         //return unit_evaluate_result * (1.0 - unit_evaluate_result); // HARDCODED: sigmoid unit
         return m_layer_neuron_function[i_layer]->evaluate_derivative(unit_dot_result);
      }
      
      /**
       *
       **/
      feed_t feed_forward( const std::vector<double>& weights 
                         , const std::vector<double>& input 
                         ) const
      {
         /////////////////////////////////////////////////////
         // for each layer feed forward and save result
         /////////////////////////////////////////////////////
         feed_t feed_forward_result;
         feed_forward_result.emplace_back(input);
         for(int i_layer = 1; i_layer < this->num_layers(); ++i_layer)
         {
            feed_forward_result.emplace_back(m_n_layer_units[i_layer]);
            for(int i_unit = 0; i_unit < m_n_layer_units[i_layer]; ++i_unit)
            {
               feed_forward_result[i_layer][i_unit] = this->unit_evaluate( i_layer
                                                                         , i_unit
                                                                         , weights
                                                                         , feed_forward_result[i_layer - 1]
                                                                         );
            }
         }
         //for(int i = 0; i < feed_forward_result.size(); ++i)
         //{
         //   std::cout << " feed_forward_result[" << i << "] = " << feed_forward_result[i] << std::endl;
         //}
         return feed_forward_result;
      }
      
      /**
       * sum-of-squares error
       **/
      double error_evaluate( const std::vector<double>& weights
                           , const training_set_t& t 
                           ) const
      {
         double error = 0.0;
         // loop over training set
         for(int t_idx = 0; t_idx < t.size(); ++t_idx)
         {
            auto& input = t[t_idx].first;
            auto& output = t[t_idx].second;
            
            auto feed_forward_result = this->feed_forward(weights, input);
            auto& output_result = feed_forward_result[this->num_layers() - 1];

            for(int i = 0; i < output.size(); ++i)
            {  
               double term_error = output[i] - output_result[i];
               error += term_error*term_error;
            }
         }
         
         error *= 0.5; // divide by two

         double sum = dot(weights, weights);
         error += m_gamma * sum;

         return error;
      }
      
      /**
       * sum-of-squares error derivative
       **/
      std::vector<double> error_derivative( const std::vector<double>& weights 
                                          , const training_set_t& t 
                                          ) const
      {
         assert_training_set(t);

         std::vector<double> derivative(weights.size(), 0.0);
         
         for(int t_idx = 0; t_idx < t.size(); ++t_idx)
         {
            auto& input = t[t_idx].first;
            auto& output = t[t_idx].second;
            
            // feed-forward
            auto feed_forward_result = this->feed_forward(weights, input);

            /////////////////////////////////////////////////////
            // for each layer calculate delta
            /////////////////////////////////////////////////////
            // out layer delta
            std::vector<std::vector<double> > layer_delta(m_n_layer_units.size());
            layer_delta[m_n_layer_units.size() - 1].resize(m_n_layer_units[m_n_layer_units.size() - 1]);
            int i_out_layer = m_n_layer_units.size() - 1;
            for(int i = 0; i < output.size(); ++i)
            {  // HARDCODED: sum-of-squares error

               //std::cout << " derivative: " << unit_derivative(i_out_layer, i, feed_forward_result[i_out_layer - 1]) << std::endl;
               layer_delta[i_out_layer][i] = (output[i] - feed_forward_result[i_out_layer][i])*this->unit_derivative(i_out_layer, i, weights, feed_forward_result[i_out_layer - 1]);
            }
            
            // hidden layer deltas
            for(int i_layer = m_n_layer_units.size() - 2; i_layer > 0; --i_layer)
            {
               int n_units = m_n_layer_units[i_layer];
               layer_delta[i_layer].resize(n_units);
               for(int i_unit = 0; i_unit < n_units; ++i_unit)
               {
                  double sum = 0.0;
                  for(int i_old_unit = 0; i_old_unit < m_n_layer_units[i_layer + 1]; ++i_old_unit)
                  {
                     sum += layer_delta[i_layer + 1][i_old_unit]*this->weight(i_layer + 1, i_old_unit, i_unit, weights);
                     //std::cout << "   layer_delta = " << layer_delta[i_layer + 1][i_old_unit];
                     //std::cout << "   weight = " << this->weight(i_layer + 1, i_old_unit, i_unit, weights);
                     //std::cout << "   sum = " << sum << std::endl;
                  }
                  //std::cout << " SUM = " << sum << std::endl;
                  layer_delta[i_layer][i_unit] = sum*this->unit_derivative(i_layer, i_unit, weights, feed_forward_result[i_layer - 1]);
               }
            }
            
            //std::cout << " **************** LAYER DELTAS ******************* " << std::endl;
            //std::cout << " weights : " << weights << std::endl;
            //for(int i = 1; i < layer_delta.size(); ++i)
            //{
            //   std::cout << " layer delta " << i << " : " << std::flush;
            //   std::cout << layer_delta[i] << std::endl;
            //}
            //std::cout << " **************** LAYER DELTAS END *************** " << std::endl;
            
            /////////////////////////////////////////////////////
            // calculate derivatives
            /////////////////////////////////////////////////////
            for(int i_layer = 1; i_layer < this->num_layers(); ++i_layer)
            {
               //std::cout << " input       = " << feed_forward_result[i_layer - 1] << std::endl;
               //std::cout << " layer_delta = " << layer_delta[i_layer] << std::endl;

               for(int i_unit = 0; i_unit < m_n_layer_units[i_layer]; ++i_unit)
               {
                  // get index pair and loop over weights! (REMEMBER bias !!)
                  auto& weight_idx_pair = m_weight_indices[i_layer][i_unit];
                  int weight_begin = weight_idx_pair.first;
                  int weight_end = weight_idx_pair.second;
                  //std::cout << " weight begin = " << weight_begin << "  weight_end = " << weight_end << std::endl;
                  
                  // NBNBNBN : sign ??
                  derivative[weight_begin] -= layer_delta[i_layer][i_unit];
                  derivative[weight_begin] += 2*m_gamma*weights[weight_begin];
                  //std::cout << " i_weight = " << weight_begin << std::endl;
                  
                  int result_idx = 0;
                  for(int i_weight = weight_begin + 1; i_weight < weight_end; ++i_weight)
                  {
                     //std::cout << " i_weight = " << i_weight << std::endl;
                     // NBNBNBN : sign ??
                     derivative[i_weight] -= layer_delta[i_layer][i_unit]*feed_forward_result[i_layer - 1][result_idx];
                     derivative[i_weight] += 2*m_gamma*weights[i_weight];
                     ++result_idx;
                  }
                  //std::cout << " DERIVATIVE DEBUG : ("; 
                  //for(int i = 0; i < derivative.size() - 1; ++i)
                  //{
                  //   std::cout << 0.2*derivative[i] << ",";
                  //}
                  //std::cout << 0.2*derivative[derivative.size() - 1] << ")" << std::endl;
               }
            }
         }

         // return derivative
         return derivative;
      }
   
   public:

};

/**
 *
 **/
class neural_network_2: private neural_network_evaluator
{
   private:
      std::vector<double> m_weights;

      bool m_optimized = false;

      /**
       *
       **/
      double random_number() const
      {
         return libmda::util::rand_float<double>()*libmda::util::random_sign<double>()/20.0;
         //return 0.05;
      }

   public:
      /**
       *
       **/
      neural_network_2
         ( const std::vector<int>& n_layer_units 
         , const std::vector<std::string>& layer_functions
         )
         : neural_network_evaluator(n_layer_units, layer_functions)
      {

         // init weights
         m_weights.resize(neural_network_evaluator::num_weights());
         for(int i = 0; i < m_weights.size(); ++i)
         {
            m_weights[i] = this->random_number();
         }

      }

      /**
       *
       **/
      std::vector<double> evaluate( std::vector<double> layer_input ) const
      {
         auto feed_forward_result = neural_network_evaluator::feed_forward(m_weights, layer_input);
         return feed_forward_result[neural_network_evaluator::num_layers() - 1];
      }


      /**
       *
       **/
      void optimize( const training_set_t& t
                   , const training_set_t& test1
                   , const training_set_t& test2
                   , int n_iter = 500
                   , double threshold = 1e-5
                   )
      {
         assert(!m_optimized);
         //evaluate_performance_single(t, *this);
         //evaluate_performance_single(test1, *this);
         //evaluate_performance_single(test2, *this);
         evaluate_performance_multiple(t, *this);
         evaluate_performance_multiple(test1, *this);
         evaluate_performance_multiple(test2, *this);

         /////////
         // initialize some stuff
         /////////
         std::vector<double> derivative = neural_network_evaluator::error_derivative(m_weights, t);
         std::vector<double> p_direction(derivative.size());
         for(int i = 0; i < p_direction.size(); ++i)
         {
            p_direction[i] = -derivative[i];
         }
         
         //int n_iter = 1;
         int i_iter = 0;
         bool converged = false;
         

         // main loop
         while( (!converged) && (i_iter < n_iter) )
         {
            //std::cout << " evaluate   = " << neural_network_evaluator::error_evaluate(m_weights, t) << std::endl;
            //std::cout << " weights    = " << m_weights << std::endl;
            //std::cout << " derivative = (";
            //for(int i = 0; i < derivative.size() - 1; ++i)
            //{
            //   std::cout << 0.2*derivative[i] << ",";
            //}
            //std::cout << 0.2*derivative[derivative.size() - 1] << ")" << std::endl;
            
            ////
            // line search
            ////
            struct network_phi_func
            {
               using step_t = double;
               using value_t = double;

               neural_network_2 const* m_p_network;
               const std::vector<double>& m_weights;
               const std::vector<double>& m_p;
               const training_set_t& m_t;

               network_phi_func( neural_network_2 const* a_p_network
                               , const std::vector<double>& a_weights
                               , const std::vector<double>& a_p
                               , const training_set_t& a_t
                               )
                  : m_p_network(a_p_network)
                  , m_weights(a_weights)
                  , m_p(a_p)
                  , m_t(a_t)
               {
               }

               double operator()(double alpha) const
               {
                  //std::cout << " evaluate ! " << std::endl;
                  //std::cout << " alpha = " << alpha << std::endl;
                  std::vector<double> argument_phi(m_weights.size());
                  for(int i = 0; i < argument_phi.size(); ++i)
                  {
                     argument_phi[i] = m_weights[i] + alpha*m_p[i];
                  }
                  auto eval = m_p_network->error_evaluate(argument_phi, m_t);
                  //std::cout << " eval = " << eval << std::endl;
                  return eval;
               }

               double first_derivative(double alpha) const
               {
                  //std::cout << " first derivative ! " << std::endl;
                  //std::cout << " alpha = " << alpha << std::endl;
                  std::vector<double> argument_phi(m_weights.size());
                  for(int i = 0; i < argument_phi.size(); ++i)
                  {
                     argument_phi[i] = m_weights[i] + alpha*m_p[i];
                  }
                  auto derivative = m_p_network->error_derivative(argument_phi, m_t);
                  //std::cout << " derivative = " << derivative << std::endl;
                  //std::cout << " mp         = " << m_p << std::endl;
                  double dot = 0.0;
                  for(int i = 0; i < derivative.size(); ++i)
                  {
                     dot += derivative[i] * m_p[i];
                  }
                  //std::cout << "dot = " << dot << std::endl;
                  return dot;
               }

            } phi(this, m_weights, p_direction, t);

            // calculate norm of p_direction
            double p_norm = 0.0;
            for(int i =0 ; i < p_direction.size(); ++i)
            {
               p_norm += p_direction[i]*p_direction[i];
            }
            p_norm = std::sqrt(p_norm);
            
            //                                                           alpha_max   c_1  c_2
            //double alpha_max = 10/p_norm;
            double alpha_max = 1000.0;
            //double alpha = libmda::numeric::optim::wolfe_line_search(phi,1.2       , 0.1 , 0.9);
            double alpha = libmda::numeric::optim::wolfe_line_search(phi, 0.0, alpha_max   , 1e-2, 1e-4, 0.1);
            //double alpha = 0.2;
           
            //////////
            // update weights
            //////////
            for(int i = 0; i < m_weights.size(); ++i)
            {
               //m_weights[i] -= alpha * derivative[i];
               m_weights[i] += alpha * p_direction[i];
            }
            
            //////////
            // calculate beta
            //////////
            double beta = 0.0;
            auto derivative_p1 = neural_network_evaluator::error_derivative(m_weights, t);
            auto beta_fr = dot(derivative_p1, derivative_p1) / dot(derivative, derivative); // flether-reeves
            auto beta_pr = (dot(derivative_p1, derivative_p1) - dot(derivative_p1, derivative) ) / dot(derivative, derivative); // polak - ribiere
            //beta_pr = std::max(0.0,beta);

            if( libmda::numeric::float_lt(beta_pr, -beta_fr) )
            {
               beta = -beta_fr;
            }
            else if( libmda::numeric::float_leq(std::abs(beta_pr), beta_fr) )
            {
               beta = beta_pr;
            }
            else if( libmda::numeric::float_gt(beta_pr, beta_fr) )
            {
               beta = beta_fr;
            }
            else
            {
               assert(false);
            }
            
            //std::cout << " derivative_p1 " << derivative_p1 << std::endl;

            //////////
            // update direction
            //////////
            for(int i = 0; i < p_direction.size(); ++i)
            {
               p_direction[i] *= beta;
               p_direction[i] -= derivative_p1[i];
            }

            /////////
            // check for convergence
            /////////
            converged = true;
            for(int i = 0; i < derivative_p1.size(); ++ i)
            {
               //std::cout << derivative_p1[i] << " " << (derivative_p1[i] > 1e-5) << std::endl;
               if(std::fabs(derivative_p1[i]) > threshold)
               {
                  //std::cout << " NOT CONVERGED ! " << std::endl;
                  converged = false;
                  break;
               }
            }

            /////////
            // output summary
            /////////
            double result = neural_network_evaluator::error_evaluate(m_weights, t);
            double norm_w = 0.0;
            double norm_p = 0.0;
            double norm_df = 0.0;
            for(int i = 0; i < m_weights.size(); ++i)
            {
               norm_w += m_weights[i]*m_weights[i];
               norm_p += p_direction[i]*p_direction[i];
               norm_df += derivative_p1[i]*derivative_p1[i];
            }
            norm_w = std::sqrt(norm_w);
            norm_p = std::sqrt(norm_p);
            norm_df = std::sqrt(norm_df);
            std::cout << " Summary iteration : " << i_iter << "\n"
                      << " f(w)              = " << result << "\n"
                      << " step              = " << alpha << "\n"
                      << " beta              = " << beta << "\n"
                      << " |w|               = " << norm_w << "\n"
                      << " |p|               = " << norm_p << "\n"
                      << " |df|              = " << norm_df << "\n";

            //evaluate_performance_single(t, *this);
            //evaluate_performance_single(test1, *this);
            //evaluate_performance_single(test2, *this);
            evaluate_performance_multiple(t, *this);
            evaluate_performance_multiple(test1, *this);
            evaluate_performance_multiple(test2, *this);
            std::cout << std::endl;
            
            /////////
            // clean-up iteration
            /////////
            //std::swap(derivative, derivative_p1);
            derivative = std::move(derivative_p1);
            ++i_iter;
         }
         
         //std::cout << " weights    = " << m_weights << std::endl;
         //std::cout << " derivative after: " << derivative << std::endl;

         m_optimized = true;
      }
};

#endif /* NEURAL_NETWORK_2_H_INCLUDED */
