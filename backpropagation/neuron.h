#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include "../../libmda/util/random_bool.h"

std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec)
{
   os << "(";
   for(int i = 0; i < vec.size()-1; ++i)
   {
      os << vec[i] << ",";
   }
   if(vec.size()>0)
   {
      os << vec[vec.size()-1];
   }
   os << ")";
   return os;
}

template<class T>
T dot(const std::vector<T>& v1, const std::vector<T>& v2)
{
   static_assert(std::is_floating_point<T>::value,"dot() only works for floating point numbers.");
   assert(v1.size() == v2.size());
   
   T dot_result = T(0.0);
   for(int i = 0; i < v1.size(); ++i)
   {
      dot_result += v1[i]*v2[i];
   }

   return dot_result;
}

/**
 *
 **/
class neuron
{
   public:
                                // eval , deriv(with a_j)
      using eval_pair = std::pair<double, double>;

   protected:
      std::vector<double> m_weights;
      double m_bias;

   private:
      /**
       *
       **/
      virtual eval_pair evaluate_impl(const double) const = 0;

      double random_number() const
      {
         //return libmda::util::rand_float<double>()*libmda::util::random_sign<double>()/10.0;
         return libmda::util::rand_float<double>()*libmda::util::random_sign<double>()/20.0;
         //return libmda::util::rand_float<double>()*libmda::util::random_sign<double>();
         //return 0.05;
      }

   public:
      neuron(int size): m_weights(size)
                      , m_bias(random_number())
      {
         for(auto& elem : m_weights) elem = random_number();
      }

      /**
       *
       **/
      eval_pair evaluate(const std::vector<double>& input) const
      {
         return evaluate_impl(dot(input,m_weights) + m_bias);
      }
      
      /**
       *
       **/
      double weight(int i) const
      {
         assert(i < m_weights.size());
         return m_weights[i];
      }

      /**
       *
       **/
      void update(const double update_bias
                 ,const std::vector<double>& update_weights
                 )
      {
         assert(update_weights.size() == m_weights.size());
         // update bias
         m_bias += update_bias;
         // update weights
         for(int i = 0; i < m_weights.size(); ++i)
         {
            m_weights[i] += update_weights[i];
         }
      }

      friend std::ostream& operator<<(std::ostream& os, const neuron& n);
};

inline std::ostream& operator<<(std::ostream& os, const neuron& n)
{
   os << "neuron: bias = " << n.m_bias << " weights = (";
   for(int i = 0; i < n.m_weights.size() - 1; ++i)
      os << n.m_weights[i] << ",";
   if(n.m_weights.size() > 0)
      os << n.m_weights[n.m_weights.size() - 1];
   os << ")";
   return os;
}

///**
// *
// **/
//class linear_unit: public neuron
//{
//   private:
//      /**
//       *
//       **/
//      double evaluate_impl(const double input) const
//      {
//         return input;
//      }
//
//   public:
//      linear_unit(int size): neuron(size)
//      {
//      }
//};
//
//
///**
// *
// **/
//class perceptron: public neuron
//{
//   private:
//      const double m_threshold = 0.5;
//      
//      /**
//       *
//       **/
//      double evaluate_impl(const double input) const
//      {
//         return input > m_threshold ? 1.0 : 0.0;
//      }
//
//   public:
//      perceptron(int size): neuron(size)
//      {
//      }
//};

/**
 *
 **/
class sigmoid: public neuron
{
   private:
      using neuron::eval_pair;

      /**
       *
       **/
      eval_pair evaluate_impl(const double input) const
      {
         auto eval = 1.0 / (1.0 + std::exp(-input));
         return {eval, eval*(1.0 - eval) };
      }

   public:
      sigmoid(int size): neuron(size)
      {
      }
};

/**
 *
 **/
class tanh_unit: public neuron
{
   private:
      using neuron::eval_pair;

      /**
       *
       **/
      eval_pair evaluate_impl(const double input) const
      {
         auto exp_plus = std::exp(input);
         auto exp_minus = std::exp(-input);
         auto eval =  (exp_plus - exp_minus) / (exp_plus + exp_minus);
         return {eval, 1.0 - eval*eval };
      }

   public:
      tanh_unit(int size): neuron(size)
      {
      }
};



#endif /* NEURON_H_INCLUDED */
