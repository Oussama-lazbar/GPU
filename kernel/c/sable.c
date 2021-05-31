#include "easypap.h"

#include <omp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned TYPE;

static TYPE *TABLE = NULL;

static volatile int changement;

static TYPE max_grains;

static inline TYPE *table_cell (TYPE *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

#define table(y, x) (*table_cell (TABLE, (y), (x)))

#define RGB(r, g, b) rgba (r, g, b, 0xFF)

void sable_init ()
{
  if (TABLE == NULL) {
    const unsigned size = DIM * DIM * sizeof (TYPE);

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    TABLE = mmap (NULL, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
}

void sable_finalize ()
{
  const unsigned size = DIM * DIM * sizeof (TYPE);

  munmap (TABLE, size);
}

///////////////////////////// Production d'une image
void sable_refresh_img ()
{
  unsigned long int max = 0;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++) {
      int g = table (i, j);
      int r, v, b;
      r = v = b = 0;
      if (g == 1)
        v = 255;
      else if (g == 2)
        b = 255;
      else if (g == 3)
        r = 255;
      else if (g == 4)
        r = v = b = 255;  // white on screen
      else if (g > 4)
        r = b = 255 - (240 * ((double)g) / (double)max_grains);

      cur_img (i, j) = RGB (r, v, b);
      if (g > max)
        max = g;
    }
  max_grains = max;
}

///////////////////////////// Version séquentielle simple (seq)

static inline int compute_new_state (int y, int x)
{
  if (table (y, x) >= 4) {
    unsigned long int div4 = table (y, x) / 4;
    table (y, x - 1) += div4;
    table (y, x + 1) += div4;
    table (y - 1, x) += div4;
    table (y + 1, x) += div4;
    table (y, x) %= 4;
    return 1;
  }
  return 0;
}

static int do_tile (int x, int y, int width, int height, int who)
{
  int chgt = 0;
  PRINT_DEBUG ('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
               y + height - 1);

  monitoring_start_tile (who);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      chgt |= compute_new_state (i, j);
    }

  monitoring_end_tile (x, y, width, height, who);
  return chgt;
}
// Separation du travail : On traite l'interieur de la tuile separement des bords 

// Pour le traitement de la partie interieure de la tuile 
static int do_inside_tile(int x , int y, int width, int height, int who)
{
  int chgt = 0;
  PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
              y + height - 1);
  monitoring_start_tile(omp_get_thread_num());

  for (int i = y + 1; i < y + height - 1; i++)
    for (int j = x + 1; j < x + width - 1; j++)
    {
      chgt |= compute_new_state(i, j);
    }

  monitoring_end_tile(x, y, width, height, omp_get_thread_num());

  return chgt;
}


// Pour le traitement des bords
static int do_boundary_tile(int x , int y, int width, int height, int who)
{
  int chgt  = 0; 
  PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
              y + height - 1);
  //monitoring_start_tile(omp_get_thread_num());

  for (int j = x ; j< x+width; j++){
    chgt |= compute_new_state(y, j);
    chgt |= compute_new_state(y + height - 1, j);
  }
  for (int i = y ; i< y + height ; i++){
    chgt |= compute_new_state(i, x);
    chgt |= compute_new_state(i, x + width - 1);
  }
 

  //monitoring_end_tile(x, y, width, height, omp_get_thread_num());

  return chgt;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned sable_compute_seq (unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it++) {
    changement = 0;
    // On traite toute l'image en un coup (oui, c'est une grosse tuile)
    changement |= do_tile (1, 1, DIM - 2, DIM - 2, 0);
    if (changement == 0)
      return it;
  }
  return 0;
}

///////////////////////////// Version séquentielle tuilée (tiled)

unsigned sable_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    changement = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        changement |= do_tile (x + (x == 0), y + (y == 0),
                               TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                               TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                               0 /* CPU id */);
    if (changement == 0)
      return it;
  }

  return 0;
}
///////////////////////////// Tiled parallel version

///////////////////////////// Version openmp tuilée (omp)

unsigned sable_compute_omp_tiled(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    changement = 0;
    // TRAITEMENT DES BORDS C'EST LA PARTIE LA PLUS GOURMANDE
    // A chaque étape on traite des bords de tuiles non voisines 
    // Pour Éviter la lecture/écriture dans une meme case 
    //  par plusieurs threads
    #pragma omp parallel
    {
      #pragma omp for collapse(2)
      for (int y = 0; y < DIM; y += 2*TILE_H)
        for (int x = 0; x < DIM; x += 2*TILE_W)
          changement |= do_boundary_tile(x + (x == 0), y + (y == 0),
                              TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                              TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                              omp_get_thread_num());

      #pragma omp for collapse(2)
      for (int y = TILE_H; y < DIM; y += 2*TILE_H)
        for (int x = 0; x < DIM; x += 2*TILE_W)
          changement |= do_boundary_tile(x + (x == 0), y + (y == 0),
                              TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                              TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                              omp_get_thread_num());

      #pragma omp for collapse(2)               
      for (int y = 0; y < DIM; y += 2*TILE_H)
        for (int x = TILE_W; x < DIM; x += 2*TILE_W)
          changement |= do_boundary_tile(x + (x == 0), y + (y == 0),
                              TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                              TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                              omp_get_thread_num());

      #pragma omp for collapse(2)
      for (int y = TILE_H; y < DIM; y += 2*TILE_H)
        for (int x = TILE_W; x < DIM; x += 2*TILE_W)
          changement |= do_boundary_tile(x + (x == 0), y + (y == 0),
                              TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                              TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                              omp_get_thread_num());

      // INTERIEUR DES TUILES : TRES PARALLELISABLE
      // c'est la partie qui  coûte le moins cher 
      // On peu utiliser un collapse(2) sans probleme 
      // puisque les interieurs des tuiles sont séparés 
      #pragma omp for collapse(2)
        for (int y = 0; y < DIM; y += TILE_H)
          for (int x = 0; x < DIM; x += TILE_W)
            changement |= do_inside_tile(x + (x == 0), y + (y == 0),
                                TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                omp_get_thread_num());
    }
        
    if (changement == 0)
      return it;
  }
  return 0;
}


////////////////  Utile pour la version GPU de base

unsigned sable_invoke_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local[2]  = {TILE_W,TILE_H};
  cl_int err;

  for (unsigned it = 1; it <= nb_iter; it++) {
    // setting first argument
    err = 0;

    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    check (err, "Setting kernel 1 failed");

    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    check (err, "Setting kernel 2 failed");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Kernel exec failed ");

    // Swap buffers
          {
            cl_mem tmp  = cur_buffer;
            cur_buffer  = next_buffer;
            next_buffer = tmp;
          }
    // Reads GPU data
        clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0,
                            sizeof(unsigned)*DIM * DIM, TABLE, 0, NULL, NULL);

  }

  clFinish (queue);

  return 0;
}


///////////////////////////// Configurations initiales

static void sable_draw_4partout (void);


void sable_draw (char *param)
{
  // Call function ${kernel}_draw_${param}, or default function (second
  // parameter) if symbol not found
  hooks_draw_helper (param, sable_draw_4partout);
}

void sable_draw_4partout (void)
{
  max_grains = 8;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      cur_img (i, j) = table (i, j) = 4;
}

void sable_draw_DIM (void)
{
  max_grains = DIM;
  for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
    for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
      cur_img (i, j) = table (i, j) = i * j / 4;
}

void sable_draw_alea (void)
{
  max_grains = 5000;
  for (int i = 0; i< DIM>>3; i++) {
    int i = 1 + random () % (DIM - 2);
    int j = 1 + random () % (DIM - 2);
    int grains = 1000 + (random () % (4000));
    cur_img (i, j) = table (i, j) = grains;
  }
}

// UNE VERSION POUR VOIR UN ÉBOULEMENT D'UN
// GRAND TAS DE SABLE INITIALEMENT AU MILIEU
void sable_draw_middle(void)
{
  max_grains = 10000;
  cur_img(DIM/2, DIM/2) = table(DIM/2, DIM/2) = max_grains;
}

