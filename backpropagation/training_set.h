#ifndef TRAINING_SET_H_INCLUDED
#define TRAINING_SET_H_INCLUDED

#include <vector>
#include <utility> // for std::pair

using input_t = std::vector<double>;
using output_t = std::vector<double>;
using training_t = std::pair<input_t, output_t>;
using training_set_t = std::vector<training_t>;

#endif /* TRAINING_SET_H_INCLUDED */
