#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include <iostream>
#include <cassert>
#include <vector>

/**
 *
 */


/**
 *
 **/
class board
{
   private:
      using piece_position_t = std::pair<int, int>;
      using board_position_t = std::vector<piece_position_t>;

      const std::vector<board_position_t> winning_positions = 
      { { {0,0}, {0,1}, {0,2} }
      , { {1,0}, {1,1}, {1,2} }
      , { {2,0}, {2,1}, {2,2} }
      , { {0,0}, {1,0}, {2,0} }
      , { {0,1}, {1,1}, {2,1} }
      , { {0,2}, {1,2}, {2,2} }
      , { {0,0}, {1,1}, {2,2} }
      , { {0,2}, {1,1}, {2,0} }
      };

   public:
      static const char x = 'x';
      static const char o = 'o';
      static const char e = ' ';
   
   private:
      static const int m_size = 3;
      char m_board[m_size][m_size];
      
      /**
       *
       **/
      void setposition(const char c, int i, int j)
      {
         assert((c == x) || (c == o) || (c == e)); // assert we set to a correct board piece
         assert((i < m_size) && (j < m_size)); // assert postion is actully in bounds
         m_board[i][j] = c;
      }

      /**
       *
       **/
      bool is_position(const char pos, int i, int j) const
      {
         return m_board[i][j] == pos;
      }

   public:
      /**
       *
       **/
      board()
      {
         for(int i = 0; i < m_size; ++i)
         {
            for(int j = 0; j < m_size; ++j)
            {
               m_board[i][j] = e;
            }
         }
      }

      /**
       *
       **/
      int check_gameover() const
      {
         bool x_won = false;
         bool o_won = false;
         for(int i = 0; i < winning_positions.size(); ++i)
         {
            auto& bp = winning_positions[i];

            bool x_won_i = true;
            for(int j = 0; j < bp.size(); ++j)
            {
               auto& i_idx = bp[j].first;
               auto& j_idx = bp[j].second;
               if(m_board[i_idx][j_idx] != x)
               {
                  x_won_i = false;
                  break;
               }
            }
            
            bool o_won_i = true;
            for(int j = 0; j < bp.size(); ++j)
            {
               auto& i_idx = bp[j].first;
               auto& j_idx = bp[j].second;
               if(m_board[i_idx][j_idx] != o)
               {
                  o_won_i = false;
                  break;
               }
            }

            x_won = x_won || x_won_i;
            o_won = o_won || o_won_i;
         }
         
         if(x_won && o_won)
         {
            std::cout << " both players won :SSSSS " << std::endl;
            assert(false);
         }

         return x_won ? 1 : (o_won ? 2 : 0);
      }

      /**
       *
       **/
      int count_piece_type(char piece)
      {
         int count = 0;
         for(int i = 0; i < m_size; ++i) 
         {
            for(int j = 0; j < m_size; ++j) 
            {
               if(m_board[i][j] == piece)
               {
                  ++count;
               }
            }
         }
         return count;
      }

      /**
       *
       **/
      bool place_piece(const char piece, int i, int j)
      {
         if(is_position(e,i,j))
         {
            setposition(piece,i,j);
            return true;
         }
         return false;
      }

      /**
       *
       **/
      bool remove_piece(const char piece, int i, int j)
      {
         //std::cout << " remove: " << piece << " " << i << "," << j << std::endl;
         if(is_position(piece,i,j))
         {
            setposition(e,i,j);
            return true;
         }
         return false;
      }

      /**
       *
       **/
      bool move_piece(const char piece, int i, int j, int k, int l)
      {
         //std::cout << "move_piece" << std::endl;
         if(remove_piece(piece,i,j))
         {
            //std::cout << " REMOVE SUCCESS " << std::endl;
            if(place_piece(piece,k,l))
            {
               return true;
            }
            place_piece(piece,i,j); // if placement fails, we place the piece from where we removed it again
         }
         return false;
      }

      /**
       *
       **/
      char player_piece_type(int player)
      {
         return (player == 1) ? x : o;
      }

      /**
       *
       **/
      friend std::ostream& operator<<(std::ostream& os, const board& b);
};

/**
 *
 **/
std::ostream& operator<<(std::ostream& os, const board& b)
{
   for(int i = 0; i < b.m_size - 1; ++i)
   {
      for(int j = 0; j < b.m_size - 1; ++j)
      {
         os << b.m_board[i][j] << "|";
      }
      os << b.m_board[i][b.m_size - 1] << "\n";
      for(int j = 0; j < b.m_size; ++j)
      {
         os << "--";
      }
      os << "\n";
   }
   // output last row
   for(int j = 0; j < b.m_size - 1; ++j)
   {
      os << b.m_board[b.m_size - 1][j] << "|";
   }
   os << b.m_board[b.m_size - 1][b.m_size - 1] << "\n";

   return os;
}

#endif /* BOARD_H_INCLUDED */
