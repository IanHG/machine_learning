#ifndef PGM_IMAGE_TRAIN_H_INCLUDED
#define PGM_IMAGE_TRAIN_H_INCLUDED

#include <map>
#include <tuple>
#include <cstring>

#include "pgm_image.h"
#include "training_set.h"

const double target_high = 0.9;
const double target_low = 0.1;

//
std::map<std::string, double > user_targets =
{ { {"at33"}, {0.025} } 
, { {"boland"}, {0.075} } 
, { {"bpm"}, {0.125} }
, { {"ch4f"}, {0.175} }
, { {"cheyer"}, {0.225} }
, { {"choon"}, {0.275} }
, { {"danieln"}, {0.325} }
, { {"glickman"}, {0.375} }
, { {"karyadi"}, {0.425} }
, { {"kawamura"}, {0.475} }
, { {"kk49"}, {0.525} }
, { {"megak"}, {0.575} }
, { {"mitchell"}, {0.625} }
, { {"night"}, {0.675} }
, { {"phoebe"}, {0.725} }
, { {"saavik"}, {0.775} }
, { {"steffi"}, {0.825} }
, { {"sz24"}, {0.875} }
, { {"an2i"}, {0.925} }
, { {"tammo"}, {0.975} }
};

//std::map<std::string, double > user_targets =
//{ { {"at33"}, {1.0} } 
//, { {"boland"}, {2.0} } 
//, { {"bpm"}, {3.0} }
//, { {"ch4f"}, {4.0} }
//, { {"cheyer"}, {5.0} }
//, { {"choon"}, {6.0} }
//, { {"danieln"}, {7.0} }
//, { {"glickman"}, {8.0} }
//, { {"karyadi"}, {9.0} }
//, { {"kawamura"}, {10.0} }
//, { {"kk49"}, {11.0} }
//, { {"megak"}, {12.0} }
//, { {"mitchell"}, {13.0} }
//, { {"night"}, {14.0} }
//, { {"phoebe"}, {15.0} }
//, { {"saavik"}, {16.0} }
//, { {"steffi"}, {17.0} }
//, { {"sz24"}, {18.0} }
//, { {"an2i"}, {19.0} }
//, { {"tammo"}, {20.0} }
//};

/**
 *
 **/
training_t pgm_to_training
   ( const pgm_image& image  
   , const double target
   )
{
   training_t train;
   train.first.resize(image.size());
   train.second.resize(1);
   for(int i = 0; i < image.size(); ++i)
   {
      train.first[i] = double(image.data(i))/255.0;
   }
   
   train.second[0] = target;

   return train;
}

/**
 *
 **/
training_t pgm_to_training_user_targets
   ( const pgm_image& image
   )
{
   int scale;
   char userid[40], head[40], expression[40], eyes[40], photo[40];

   userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';
   
   // scan in the image features
   sscanf( image.name().c_str()
         , "%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]"
         , userid, head, expression, eyes, &scale, photo
         );
   
   // compare
   return pgm_to_training(image, user_targets[std::string(userid)]);
}

/**
 *
 **/
training_t pgm_to_training_user( const pgm_image& image
                               , const std::string& user
                               )
{
   int scale;
   char userid[40], head[40], expression[40], eyes[40], photo[40];

   userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

   // scan in the image features
   sscanf( image.name().c_str()
         , "%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]"
         , userid, head, expression, eyes, &scale, photo
         );
   
   // compare
   if (!strcmp(userid, user.c_str())) 
   {
      return pgm_to_training(image, target_high);
   } 
   else 
   {
      return pgm_to_training(image, target_low);
   }
}

/**
 *
 **/
training_t pgm_to_training_eyes( const pgm_image& image
                               , const std::string& user
                               )
{
   int scale;
   char userid[40], head[40], expression[40], eyes[40], photo[40];

   userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

   // scan in the image features
   sscanf( image.name().c_str()
         , "%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]"
         , userid, head, expression, eyes, &scale, photo
         );
   
   // compare
   if (!strcmp(eyes, user.c_str())) 
   {
      return pgm_to_training(image, target_high);
   } 
   else 
   {
      return pgm_to_training(image, target_low);
   }
}

/**
 *
 **/
training_set_t pgm_to_training_set_user_targets
   ( const std::vector<pgm_image>& image_list
   )
{
   training_set_t train;
   for(int i = 0; i < image_list.size(); ++i)
   {
      train.emplace_back(pgm_to_training_user_targets(image_list[i]));
   }
   return train;
}

/**
 *
 **/
training_set_t pgm_to_training_set_user( const std::vector<pgm_image>& image_list
                                       )
{
   training_set_t train;
   std::string user = "glickman";
   for(int i = 0; i < image_list.size(); ++i)
   {
      train.emplace_back(pgm_to_training_user(image_list[i], user));
   }
   return train;
}

/**
 *
 **/
training_set_t pgm_to_training_set_eyes( const std::vector<pgm_image>& image_list
                                       )
{
   training_set_t train;
   std::string user = "sunglasses";
   for(int i = 0; i < image_list.size(); ++i)
   {
      train.emplace_back(pgm_to_training_eyes(image_list[i], user));
   }
   return train;
}

/**
 *
 **/
template<class T>
void evaluate_performance_single
   ( const training_set_t& train
   , const T& net
   )
{
   int n_correct = 0;
   int n_error = 0;
   double error_avg = 0.0;
   
   // loop over training data
   for(int i_train = 0; i_train < train.size(); ++i_train)
   {
      auto& input = train[i_train].first;  
      auto& output = train[i_train].second;
      std::vector<double> error(output.size());

      auto eval = net.evaluate(input);
         
      bool correct = true;
      for(int i = 0; i < output.size(); ++i)
      {
         auto delta = output[i] - eval[i];
         error[i] = 0.5*delta*delta;
         error_avg += error[i];
         ++n_error;
         
         if(output[i] > 0.5)
         {
            if(eval[i] < 0.5)
            {
               correct = false;
            }
         }
         else
         {
            if(eval[i] > 0.5)
            {
               correct = false;
            }
         }
      }
      if(correct)
      {
         ++n_correct;
      }
   }
   
   error_avg = n_error > 0 ? error_avg / double(n_error) : 0.0;
   auto percent = train.size() > 0 ? double(n_correct) / double(train.size())*100.0 : 0.0;

   std::cout << " error avg : " << error_avg << "   correct : " << n_correct << " (" << percent << "%)" << std::endl;
}

/**
 *
 **/
template<class T>
void evaluate_performance_multiple
   ( const training_set_t& train
   , const T& net
   )
{
   int n_correct = 0;
   int n_error = 0;
   double error_avg = 0.0;
   
   // loop over training data
   for(int i_train = 0; i_train < train.size(); ++i_train)
   {
      auto& input = train[i_train].first;  
      auto& output = train[i_train].second;
      std::vector<double> error(output.size());

      auto eval = net.evaluate(input);
         
      bool correct = true;
      for(int i = 0; i < output.size(); ++i)
      {
         auto delta = output[i] - eval[i];
         error[i] = 0.5*delta*delta;
         error_avg += error[i];
         ++n_error;
         
         if(eval[i] < (output[i] - 0.025))
         //if(eval[i] < (output[i] - 0.5))
         {
            correct = false;
         }
         else if(eval[i] > (output[i] + 0.025))
         //else if(eval[i] > (output[i] + 0.5))
         {
            correct = false;
         }
      }
      if(correct)
      {
         ++n_correct;
      }
   }
   
   error_avg = n_error > 0 ? error_avg / double(n_error) : 0.0;
   auto percent = train.size() > 0 ? double(n_correct) / double(train.size())*100.0 : 0.0;

   std::cout << " error avg : " << error_avg << "   correct : " << n_correct << " (" << percent << "%)" << std::endl;
}

#endif /* PGM_IMAGE_TRAIN_H_INCLUDED */
