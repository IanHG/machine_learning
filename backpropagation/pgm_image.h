#ifndef PGM_IMAGE_H_INCLUDED
#define PGM_IMAGE_H_INCLUDED

#include <string>
#include <stdio.h>
#include <memory> // for std::unique_ptr
#include <vector>


/**
 *
 **/
class pgm_image
{
   private:
      std::string m_name;
      int m_rows;
      int m_cols;
      int m_max_val;
      std::unique_ptr<int[]> m_data;
      
      /**
       *
       **/
      std::string base_name(const std::string& name) const
      {
         auto found = name.find_last_of("/");
         return name.substr(found+1);
      }

   public:
      /**
       * ctor
       **/
      pgm_image( const std::string& name 
               , int rows = 0
               , int cols = 0
               , int max_val = 0
               )
         : m_name(base_name(name))
         , m_rows(rows)
         , m_cols(cols)
         , m_max_val(max_val)
         , m_data(new int[m_rows*m_cols])
      {
      }

      /**
       *
       **/
      int size() const { return m_rows * m_cols; }

      /**
       *
       **/
      const std::string& name() const { return m_name; }

      /**
       *
       **/
      int&       data(const int i)       { assert(i < size()); return m_data[i]; }
      const int& data(const int i) const { assert(i < size()); return m_data[i]; }
      
      /**
       *
       **/
      void dump() const
      {
         // implement me
         assert(false);
      }

      /**
       *
       **/
      friend std::ostream& operator<<(std::ostream&, const pgm_image&);
      friend pgm_image pgm_open(const std::string&);
};

/**
 *
 **/
std::ostream& operator<<(std::ostream& os, const pgm_image& image)
{
   os << " pgm image :  " << image.m_rows << " x " << image.m_cols << "    maxval = " << image.m_max_val << "\n";
   int counter = 0;
   for(int i = 0; i < image.m_rows; ++i)
   {
      for(int j = 0; j < image.m_cols; ++j)
      {
         os << image.m_data[counter] << " ";
         ++counter;
      }
   }
   return os;
}

/**
 * 
 **/
std::ostream& operator<<(std::ostream& os, const std::vector<pgm_image>& image_vec)
{
   for(int i = 0; i < image_vec.size(); ++i)
   {
      os << "[" << i << "]  " << image_vec[i] << "\n";
   }
   return os;
}

/**
 *
 **/
pgm_image pgm_open(const std::string& filename)
{
   //
   // open file
   //
   FILE* pgm_file;
   if( !(pgm_file = fopen(filename.c_str(), "r")) )
   {
      std::cout << "PGMOPEN: Couldn't open '" << filename << "'" << std::endl;
      return pgm_image(filename); // return default
   }
   
   //
   // Scan pnm type information, expecting P5
   //
   char line[512];
   fgets(line, 511, pgm_file);
   int type;
   sscanf(line, "P%d", &type);
   if (type != 5 && type != 2) 
   {
      std::cout << "PGMOPEN: Only handles pgm files (type P5 or P2)" << std::endl;
      fclose(pgm_file);
      return pgm_image(filename);
   }
   
   // 
   // Get dimensions of pgm
   //
   fgets(line, 511, pgm_file);
   int nc, nr; // hold number of rows and cols
   sscanf(line, "%d %d", &nc, &nr);

   //
   // Get maxval
   //
   fgets(line, 511, pgm_file);
   int max_val;
   sscanf(line, "%d", &max_val);
   if (max_val > 255) 
   {
      std::cout << "PGMOPEN: Only handles pgm files of 8 bits or less" << std::endl;
      fclose(pgm_file);
      return pgm_image(filename);
   }

   //
   // Allocate image
   //
   pgm_image image(filename, nr, nc, max_val);
   int counter = 0;
   if(type == 5) // PG5
   {
      for(int i_rows = 0; i_rows < nr; ++i_rows)
      {
         for(int i_cols = 0; i_cols < nc; ++i_cols)
         {
            image.m_data[counter] = fgetc(pgm_file);
            ++counter;
         }
      }
   }
   else if(type == 2) // PG2
   {
      char ch;
      char intbuf[100];
      int k, found;
      for(int i_rows = 0; i_rows < nr; ++i_rows)
      {
         for(int i_cols = 0; i_cols < nc; ++i_cols)
         {
            k = 0;
            found = 0;
            while (!found) 
            {
               ch = (char) fgetc(pgm_file);
               if (ch >= '0' && ch <= '9') 
               {
                  intbuf[k] = ch;  
                  ++k;
               } 
               else 
               {
                  if (k != 0) 
                  {
                     intbuf[k] = '\0';
                     found = 1;
                  }
               }
            }

            image.m_data[counter] = atoi(intbuf);
            ++counter;
         }
      }
   }
   else
   {
      std::cout << "PGMOPEN: Fatal impossible error" << std::endl;
      fclose(pgm_file);
      return image;
   }

   //
   // clean up
   //
   fclose(pgm_file);
   return image;
}

/**
 *
 **/
std::vector<pgm_image> pgm_open_from_textfile(const std::string& filename)
{
   std::vector<pgm_image> pgm_list;
   FILE* fp;
   char buf[2000];

   // open file
   if( !(fp = fopen(filename.c_str(),"r")) )
   {
      std::cout << " PGM_OPEN_FROM_TEXTFILE: Couldn't open file '" << filename << "'" << std::endl;
      return pgm_list;
   }

   // open all images
   while( fgets(buf, 1999, fp) )
   {
      int j = 0;
      while( buf[j] != '\n') ++j;
      buf[j] = '\0';
      pgm_list.emplace_back(pgm_open(std::string(buf)));
   }
   
   // cleanup
   fclose(fp);
   return pgm_list;
}

#endif /* PGM_IMAGE_H_INCLUDED */
