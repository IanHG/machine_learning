#ifndef NEURONFUNCTION_H_INCLUDED
#define NEURONFUNCTION_H_INCLUDED

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>


/**
 *
 **/
class neuron_function
{
   private:
      /**
       * 
       **/
      virtual double evaluate_impl(const double input_dot) const = 0;
      /**
       *
       **/
      virtual double evaluate_derivative_impl(const double input_dot) const = 0;
      /**
       *
       **/
      virtual std::string type_impl() const = 0;

   public:
      /**
       *
       **/
      neuron_function()
      {
      }

      /**
       * evaluate interface
       * @input_dot input
       **/
      double evaluate(const double input_dot) const
      {
         return evaluate_impl(input_dot);
      }

      /**
       * evaluate derivative interface
       * @input_dot input
       **/
      double evaluate_derivative(const double input_dot) const
      {
         return evaluate_derivative_impl(input_dot);
      }

      /**
       * get type of neuron function
       **/
      std::string type() const
      {
         return type_impl();
      }

      /**
       *
       **/
      static std::unique_ptr<neuron_function> factory(const std::string&);
      
      /**
       * operator output overload
       **/
      friend std::ostream& operator<<(std::ostream& os, const neuron_function& n);
};

/**
 *
 **/
inline std::ostream& operator<<(std::ostream& os, const neuron_function& n)
{
   os << " neuron type : " << n.type() << std::endl;
   return os;
}

/**
 *  *
 *   **/
class null_function
   : public neuron_function
{
   private:
      /**
       *
       **/
      double evaluate_impl( const double input ) const override
      {
         std::cout << " calling null function evaluate_impl() " << std::endl;
         assert(false);
         return 0.0;
      }

      /**
       *
       **/
      double evaluate_derivative_impl( const double input ) const override
      {
         std::cout << " calling null function evaluate_derivative_impl() " << std::endl;
         assert(false);
         return 0.0;
      }

      /**
       *
       **/
      std::string type_impl() const override
      {
         return {"null_function"};
      }

   public:
      /**
       *
       **/
      null_function()
         : neuron_function()
      {
      }
};

/**
 *
 **/
class linear_function
   : public neuron_function
{
   private:
      /**
       *
       **/
      double evaluate_impl( const double input ) const override
      {
         return input;
      }
      
      /**
       *
       **/
      double evaluate_derivative_impl( const double input ) const override
      {
         return 1.0;
      }

      /**
       *
       **/
      std::string type_impl() const override
      {
         return {"linear_function"};
      }

   public:
      /**
       *
       **/
      linear_function()
         : neuron_function()
      {
      }
};

/**
 *
 **/
class sigmoid_function
   : public neuron_function
{
   private:
      /**
       *
       **/
      double evaluate_impl( const double input_dot ) const override
      {
         auto eval = 1.0 / (1.0 + std::exp(-input_dot));
         return eval;
      }

      /**
       *
       **/
      double evaluate_derivative_impl( const double input_dot ) const override
      {
         auto eval = evaluate_impl(input_dot);
         return eval*(1.0 - eval);
      }

      /**
       *
       **/
      std::string type_impl() const override
      {
         return {"sigmoid_function"};
      }


   public:
      sigmoid_function()
         : neuron_function()
      {
      }
};

/**
 *
 **/
class tanh_function
   : public neuron_function
{
   private:
      /**
       *
       **/
      double evaluate_impl( const double input_dot ) const override
      {
         auto exp_plus = std::exp(input_dot);
         auto exp_minus = std::exp(-input_dot);
         auto eval =  (exp_plus - exp_minus) / (exp_plus + exp_minus);
         return eval;
      }
      
      /**
       *
       **/
      double evaluate_derivative_impl( const double input_dot ) const override
      {
         auto eval = evaluate_impl(input_dot);
         return 1.0 - eval*eval;
      }

      /**
       *
       **/
      std::string type_impl() const override
      {
         return {"tanh_function"};
      }

   public:
      tanh_function()
         : neuron_function()
      {
      }
};

/**
 *
 **/
class gaussian_function
   : public neuron_function
{
   double m_alpha = 1.0;

   private:
      /**
       *
       **/
      double evaluate_impl( const double input_dot ) const override
      {
         auto eval = std::exp(-m_alpha*input_dot*input_dot);
         return eval;
      }
      
      /**
       *
       **/
      double evaluate_derivative_impl( const double input_dot ) const override
      {
         auto eval = evaluate_impl(input_dot);
         return -m_alpha*input_dot*eval;
      }

      /**
       *
       **/
      std::string type_impl() const override
      {
         return {"gaussian_function"};
      }

   public:
      gaussian_function()
         : neuron_function()
      {
      }
};

/**
 *
 **/
inline std::unique_ptr<neuron_function> neuron_function::factory(const std::string& function)
{
   if(function == "linear")
   {
      return std::unique_ptr<neuron_function>(new linear_function);
   }
   else if(function == "sigmoid")
   {
      return std::unique_ptr<neuron_function>(new sigmoid_function);
   }
   else if(function == "tanh")
   {
      return std::unique_ptr<neuron_function>(new tanh_function);
   }
   else if(function == "gaussian")
   {
      return std::unique_ptr<neuron_function>(new gaussian_function);
   }
   else if(function == "null")
   {
      return std::unique_ptr<neuron_function>(new null_function);
   }
   else
   {
      std::cout << " unknown function type : " << function << std::endl;
      assert(false);
   }
   return {nullptr};
}

#endif /* NEURONFUNCTION_H_INCLUDED */
