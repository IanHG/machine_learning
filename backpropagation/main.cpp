#include <iostream>
#include <iomanip> // for std setprecision
#include <vector>
#include <utility> // for std pair
#include <tuple> // for std tie

#include "neuron.h"
#include "neural_network.h"
#include "neural_network_2.h"
#include "training_set.h"
#include "pgm_image_train.h"
#include "pgm_image.h"

int main()
{
   std::cout << std::setprecision(16);
   std::cout << std::scientific;

   //training_set_t train = { { {1,0,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0} }
   //                       , { {0,1,0,0,0,0,0,0}, {0,1,0,0,0,0,0,0} } 
   //                       , { {0,0,1,0,0,0,0,0}, {0,0,1,0,0,0,0,0} } 
   //                       , { {0,0,0,1,0,0,0,0}, {0,0,0,1,0,0,0,0} } 
   //                       , { {0,0,0,0,1,0,0,0}, {0,0,0,0,1,0,0,0} } 
   //                       , { {0,0,0,0,0,1,0,0}, {0,0,0,0,0,1,0,0} } 
   //                       , { {0,0,0,0,0,0,1,0}, {0,0,0,0,0,0,1,0} } 
   //                       , { {0,0,0,0,0,0,0,1}, {0,0,0,0,0,0,0,1} } 
   //                       //, { {1,1,0,0,0,0,0,0}, {1,1,0,0,0,0,0,0} } 
   //                       //, { {0,0,1,1,0,0,0,0}, {0,0,1,1,0,0,0,0} } 
   //                       //, { {0,1,1,0,0,0,0,0}, {0,1,1,0,0,0,0,0} } 
   //                       };

   //neural_network_2 network_2({8,3,8});
   //network_2.optimize(train,5000,1e-6);
   //std::vector<double> inp = {1,0,0,0,0,0,0,0};
   //std::vector<double> inp2 = {1,0,0,0,0,0,0,1};
   //std::vector<double> inp = {1,1,1};
   
   //neural_network network_test(3,3,2);
   //neural_network_2 network_2({3,2,3});
   //std::cout << "TEST evaluate " << std::endl;
   //std::cout << network_test.evaluate(inp) << std::endl;
   //std::cout << network_2.evaluate(inp) << std::endl;

   //training_set_t train_small = { { {1,1,1}, {1,1,1} } };
   //network_2.optimize(train_small, 5000);

   //auto network = backpropagation(train, 0.2, 5000, 8, 8, 3);
   //auto network = backpropagation(train_small, 0.2, 5000, 3, 3, 2);
   
   //std::cout << " ****************** INPUT 1: ***************** " << std::endl;
   //std::cout << network.evaluate(inp) << std::endl;
   //std::cout << network_2.evaluate(inp) << std::endl;
   //std::cout << " ****************** INPUT 2: ***************** " << std::endl;
   //std::cout << network.evaluate(inp2) << std::endl;
   //std::cout << network_2.evaluate(inp2) << std::endl;

   //auto pgm_list = pgm_open_from_textfile("straightrnd_train.list");
   auto pgm_list = pgm_open_from_textfile("all_train.list");
   auto train = pgm_to_training_set_user_targets(pgm_list);
   
   //auto pgm_list_test1 = pgm_open_from_textfile("straightrnd_test1.list");
   auto pgm_list_test1 = pgm_open_from_textfile("all_test1.list");
   auto test1 = pgm_to_training_set_user_targets(pgm_list_test1);
   
   //auto pgm_list_test2 = pgm_open_from_textfile("straightrnd_test2.list");
   auto pgm_list_test2 = pgm_open_from_textfile("all_test2.list");
   auto test2 = pgm_to_training_set_user_targets(pgm_list_test2);
   
   auto network = backpropagation(train, test1, test2, 0.2, 75, 960, 1, 20);
   
   neural_network_2 network_2({960,20,1},{"tanh","sigmoid"});
   network_2.optimize(train, test1, test2, 1000);
   std::cout << " ***************** NETWORK 2 ********************** " << std::endl;
   //evaluate_performance_single(train, network_2);
   //evaluate_performance_single(test1, network_2);
   //evaluate_performance_single(test2, network_2);
   evaluate_performance_multiple(train, network_2);
   evaluate_performance_multiple(test1, network_2);
   evaluate_performance_multiple(test2, network_2);
   std::cout << " ***************** NETWORK 1 ********************** " << std::endl;
   //evaluate_performance_single(train, network);
   //evaluate_performance_single(test1, network);
   //evaluate_performance_single(test2, network);
   evaluate_performance_multiple(train, network);
   evaluate_performance_multiple(test1, network);
   evaluate_performance_multiple(test2, network);

   std::cout << "EVAL: " << network_2.evaluate(test2[0].first) << " " << test2[0].second << std::endl;
   std::cout << "EVAL: " << network_2.evaluate(test2[1].first) << " " << test2[1].second << std::endl;

   return 0;
}
