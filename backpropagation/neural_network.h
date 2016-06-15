#ifndef NEURAL_NETWORK_H_INCLUDED
#define NEURAL_NETWORK_H_INCLUDED

#include <vector>

#include "neuron.h"
#include "training_set.h"
#include "pgm_image_train.h"

class neural_network
{
   private:
      int m_n_in, m_n_out, m_n_hidden;
      std::vector<sigmoid> m_out;
      std::vector<sigmoid> m_hidden;
      //std::vector<tanh_unit> m_out;
      //std::vector<tanh_unit> m_hidden;

   public:
      /**
       *
       **/
      neural_network(int n_in, int n_out, int n_hidden)
         : m_n_in(n_in)
         , m_n_out(n_out)
         , m_n_hidden(n_hidden)
         , m_out(n_out, n_hidden)
         , m_hidden(n_hidden, n_in)
      {
      }
      
      /**
       *
       **/
      std::vector<double> evaluate(const std::vector<double>& input) const
      {
         assert(input.size() == m_n_in); // assert input has correct size

         std::vector<double> hidden_result(m_n_hidden);
         for(int i = 0; i < m_n_hidden; ++i)
         {
            hidden_result[i] = m_hidden[i].evaluate(input).first;
         }

         std::vector<double> out_result(m_n_out);
         for(int i = 0; i < m_n_out; ++i)
         {
            out_result[i] = m_out[i].evaluate(hidden_result).first;
         }

         return out_result;
      }
      
      /**
       *
       **/
      //std::vector<tanh_unit>& out() { return m_out; }
      //std::vector<tanh_unit>& hidden() { return m_hidden; }
      std::vector<sigmoid>& out() { return m_out; }
      std::vector<sigmoid>& hidden() { return m_hidden; }
};

/**
 *
 **/
neural_network backpropagation
   ( const training_set_t& t
   , const training_set_t& test1
   , const training_set_t& test2
   , const double learning_rate
   , const int n_epoch
   , int n_in
   , int n_out
   , int n_hidden
   )
{
   neural_network network(n_in, n_out, n_hidden);

   auto& out = network.out();
   auto& hidden = network.hidden();
   
   // do some allocations
   std::vector<double> update_weights(std::max(n_in, n_hidden));
   
   std::vector<double> hidden_result(n_hidden);
   std::vector<double> hidden_deriv(n_hidden);
   
   std::vector<double> out_result(n_out);
   std::vector<double> out_deriv(n_out);

   std::vector<double> out_delta(n_out);
   std::vector<double> hidden_delta(n_hidden);

   // loop over training epochs
   for(int epoch = 0; epoch < n_epoch; ++epoch)
   {
      //std::cout << " HIDDEN UNITS: " << std::endl;
      //for(int i = 0; i < hidden.size(); ++i)
      //{
      //   std::cout << hidden[i] << std::endl;
      //}
      //std::cout << std::endl;
      //std::cout << " OUT UNITS: " << std::endl;
      //for(int i = 0; i < out.size(); ++i)
      //{
      //   std::cout << out[i] << std::endl;
      //}
      //std::cout << std::endl;

      // loop over training data
      for(int t_idx = 0; t_idx < t.size(); ++t_idx)
      {  
         auto& input = t[t_idx].first;
         auto& output = t[t_idx].second;

         //////////////
         // evaluate output
         //////////////
         // evaluate hidden
         for(int i = 0; i < hidden.size(); ++i)
         {
            std::tie(hidden_result[i], hidden_deriv[i]) = hidden[i].evaluate(input);
         }
         // evaluate out
         for(int i = 0; i < out.size(); ++i)
         {
            std::tie(out_result[i], out_deriv[i]) = out[i].evaluate(hidden_result);
         }
         
         //////////////
         // evaluate delta
         //////////////
         // evaluate out delta
         for(int i = 0; i < out.size(); ++i)
         {
            out_delta[i] = out_deriv[i]*(output[i] - out_result[i]);
         }
         // evaluate hidden delta
         for(int i = 0; i < hidden.size(); ++i)
         {
            double sum = 0.0;
            for(int j = 0; j < out.size(); ++j)
            {
               sum += out[j].weight(i)*out_delta[j];
               //std::cout << "   out_delta = " << out_delta[j]
               //          << "   weight = "    << out[j].weight(i)
               //          << "   sum = "       << sum << std::endl;
            }
            //std::cout << " sum = " << sum << std::endl;
            hidden_delta[i] = hidden_deriv[i] * sum;
         }
         //std::cout << " =============== LAYER DELTA 2 ================ " << std::endl;
         //std::cout << " hidden delta : " << hidden_delta << std::endl;
         //std::cout << " out delta    : " << out_delta << std::endl;
         //std::cout << " =============== LAYER DELTA 2 END ============ " << std::endl;

         //////////////
         // update weights
         //////////////
         // update hidden
         for(int i = 0; i < hidden.size(); ++i)
         {
            update_weights.resize(n_in);
            for(int j = 0; j < update_weights.size(); ++j)
            {
               update_weights[j] = learning_rate * hidden_delta[i] * input[j];
            }
            double update_bias = learning_rate * hidden_delta[i];
            //std::cout << update_bias << std::endl;
            //std::cout << update_weights << std::endl;
            hidden[i].update(update_bias, update_weights);
         }
         // update out
         for(int i = 0; i < out.size(); ++i)
         {
            update_weights.resize(n_hidden);
            for(int j = 0; j < update_weights.size(); ++j)
            {
               update_weights[j] = learning_rate * out_delta[i] * hidden_result[j];
            }
            double update_bias = learning_rate * out_delta[i];
            //std::cout << update_bias << std::endl;
            //std::cout << update_weights << std::endl;
            out[i].update(update_bias, update_weights);
         }
      }
      
      std::cout << " iteration " << epoch << std::endl;
      //evaluate_performance_single(t, network);
      //evaluate_performance_single(test1, network);
      //evaluate_performance_single(test2, network);
      evaluate_performance_multiple(t, network);
      evaluate_performance_multiple(test1, network);
      evaluate_performance_multiple(test2, network);
   }
   
   // output code
   ///std::cout << " HIDDEN UNITS: " << std::endl;
   ///for(int i = 0; i < hidden.size(); ++i)
   ///{
   ///   std::cout << hidden[i] << std::endl;
   ///}
   ///std::cout << std::endl;
   ///std::cout << " OUT UNITS: " << std::endl;
   ///for(int i = 0; i < out.size(); ++i)
   ///{
   ///   std::cout << out[i] << std::endl;
   ///}
   ///std::cout << std::endl;
   ///
   ///for(int i = 0; i < t.size(); ++i)
   ///{
   ///   auto& input = t[i].first;
   ///   auto& output = t[i].second;

   ///   for(int j = 0; j < hidden.size(); ++j)
   ///   {
   ///      std::tie(hidden_result[j], hidden_deriv[j]) = hidden[j].evaluate(input);
   ///   }
   ///   for(int j = 0; j < out.size(); ++j)
   ///   {
   ///      std::tie(out_result[j], out_deriv[j]) = out[j].evaluate(hidden_result);
   ///   }
   ///   std::cout << input << " -> " << hidden_result << " -> " << out_result << " [" << output << "]" << std::endl;
   ///}
   ///std::cout << std::endl;

   return network;
}

#endif /* NEURAL_NETWORK_H_INCLUDED */
