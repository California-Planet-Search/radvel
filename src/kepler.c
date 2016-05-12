#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int sign(int x) {
  return (x > 0) - (x < 0);
}

/*
Solution to Kepler's equation. Given mean anomally, M, and eccentricity, e, 
solve for E, the eccentric anomally, which must satisfy:

    E - e sin(E) - M = 0

Follows the method of Danby 1988 as written in Murray and Dermot p36-37.
*/

double kepler(double M, double e);
double kepler(double M, double e)
{
  int MAX_ITER = 30;
  double CONV_TOL = 1.0e-12; // convergence criterion

  double k, E, fi, d1, fip, fipp, fippp;
  int count;
  k = 0.85; // initial guess at input parameter
  count = 0; // how many loops have we done?

  E = M + sign(sin(M)) * k * e; // first guess at E, the eccentric anomally

  // E - e * sin(E) - M should go to 0
  fi = (E - e * sin(E) - M); 
  while ( fabs(fi) > CONV_TOL && count < MAX_ITER)
    {
      count++;
      
      // first, second, and third order derivatives of fi with respect to E
      fip = 1 - e * cos(E) ;
      fipp = e * sin(E); 
      fippp = 1 - fip; 

      // first, second, and third order corrections to E
      d1 = -fi / fip; 
      d1 = -fi / (fip + d1 * fipp / 2.0); 
      d1 = -fi / (fip + d1 * fipp / 2.0 + d1 * d1 * fippp / 6.0);
      E += d1;

      fi  = (E - e * sin(E) - M);
      // printf("E =  %f, count = %i\n", E , count); //debugging

      if(count==MAX_ITER){
	printf("Error: kepler step not converging after %d steps.\n", MAX_ITER);
	printf("E=%f,  M=%f,  e=%f\n", E, M, e);
	exit(-1);
      }
    }
  return E;
}

void kepler_array(double * M_array, double e, double * E_array, int size)
{
  int i;
  for(i=0;i<size;i++){
    E_array[i] = kepler(M_array[i], e);
  }
}

// little test function 
int main()
{
  double M, e, E;
  printf("Enter M, the mean anomally\n");
  scanf("%lf", &M);
  printf("Enter e, the eccentricity\n");
  scanf("%lf", &e);
  E = kepler(M, e);
  printf("%lf\n",E);
  return 0;
}
