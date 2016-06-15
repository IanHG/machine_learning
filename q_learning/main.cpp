#include <iostream>
#include <string>

#include "board.h"

std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec)
{
   for(auto& elem : vec)
   {
      os << elem << " ";
   }
   return os;
}

int next_player(int player)
{
   return ((player == 1) ? 2 : 1);
}

std::vector<std::string> cut_string(const std::string& str, char delim)
{
   std::vector<std::string> str_vec;
   int last = 0;
   int i = 0;
   for(; i < str.size(); ++i)
   {
      if(str[i] == delim)
      {
         str_vec.emplace_back(str.substr(last,i));
         last = i+1;
      }
   }
   str_vec.emplace_back(str.substr(last,i));
   return str_vec;
}

int main()
{
   board b;
   
   bool gameover = false;
   int player = 1;
   std::string move_str;
   size_t end;

   // game loop
   while(!gameover)
   {
      std::cout << " PLAYER " << player << "'s turn " << std::endl;
      std::cout << b << std::endl;
      
      bool validmove = false;
      while(!validmove)
      {
         char piece_type = b.player_piece_type(player);
         
         std::cout << " type in move " << std::endl;
         std::cin >> move_str;
         
         int n_piece = b.count_piece_type(piece_type);
         if(n_piece == 3)
         {
            auto move_vec = cut_string(move_str,'-');
            if(move_vec.size() != 2) continue;
            auto from_vec = cut_string(move_vec[0],',');
            auto to_vec = cut_string(move_vec[1],',');
            if(from_vec.size() != 2 || to_vec.size() != 2) continue;
            if(from_vec[0] == to_vec[0] && from_vec[1] == to_vec[1]) continue;
            
            if(b.move_piece( piece_type
                           , std::stoi(from_vec[0],&end), std::stoi(from_vec[1],&end)
                           , std::stoi(to_vec[0],&end), std::stoi(to_vec[1],&end)
                           )
              )
            {
               validmove = true;
            }

         }
         else if(n_piece < 3)
         {
            auto from_vec = cut_string(move_str,',');
            if(from_vec.size() != 2) continue;
            if(b.place_piece( piece_type
                            , std::stoi(from_vec[0],&end), std::stoi(from_vec[1],&end)
                            )
              )
            {
               validmove = true;
            }
         }
         else
         {
            std::cout << " you cheater " << std::endl;
            gameover = true;
         }
      }

      switch(b.check_gameover())
      {
         case 1:
            std::cout << "X won" << std::endl;
            gameover = true;
            break; // break case
         case 2:
            std::cout << "O won" << std::endl;
            gameover = true;
            break; // break case
      }

      player = next_player(player);
   }

   std::cout << b << std::endl;

   return 0;
}
