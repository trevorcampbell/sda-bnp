#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <limits.h>
#include "matching.hpp"

HungarianMatching::HungarianMatching(MXi costs){
	this->costs = costs;
}


int HungarianMatching::run(std::vector<int>& matchings) {
  int i, j, k, l, t, q, unmatched, s, cost;

  int* col_mate;
  int* row_mate;
  int* parent_row;
  int* unchosen_row;
  int s, cost;
  int* row_dec;
  int* col_inc;
  int* slack;
  int* slack_row;

  cost=0;

  //init memory to 0
  col_mate = (int*)calloc(n, sizeof(int));
  unchosen_row = (int*)calloc(n, sizeof(int));
  row_dec  = (int*)calloc(n, sizeof(int));
  slack_row  = (int*)calloc(n, sizeof(int));
  row_mate = (int*)calloc(n, sizeof(int));
  parent_row = (int*)calloc(n, sizeof(int));
  col_inc = (int*)calloc(n, sizeof(int));
  slack = (int*)calloc(n, sizeof(int));


  // Begin subtract column minima in order to start with lots of zeroes 12
  //printf("Using heuristic\n");
  for (l=0;l<n;l++)
  {
    s=costs(0, l);
    for (k=1;k<n;k++) 
      if (costs(k, l)<s)
        s=costs(k, l);
    cost+=s;
    if (s!=0)
      for (k=0;k<n;k++)
        costs(k, l)-=s;
  }
  // End subtract column minima in order to start with lots of zeroes 12

  // Begin initial state 16
  t=0;
  for (l=0;l<n;l++)
  {
    row_mate[l]= -1;
    parent_row[l]= -1;
    col_inc[l]=0;
    slack[l]=INT_MAX; //INFINITY;
  }
  for (k=0;k<n;k++)
  {
    s=costs(k, 0);
    for (l=1;l<n;l++)
      if (costs(k, l)<s)
        s=costs(k, l);
    row_dec[k]=s;
    bool chosen_row_flag = false;
    for (l=0;l<n;l++)
      if (s == costs(k, l) && row_mate[l]<0)
      {
        col_mate[k]=l;
        row_mate[l]=k;
        //if(n>=24) printf("matching col %d==row %d\n",l,k);
	chosen_row_flag = true;
	break;
      }
    if(!chosen_row_flag){
    	col_mate[k]= -1;
    	unchosen_row[t++]=k;
    	    //if(n>=24) printf( "node %d: unmatched row %d\n",t,k);
    }
  }
  // End initial state 16

  // Begin Hungarian algorithm 18
  unmatched=t;
  if(t > 0){
  while (1){
    //if(n>=24) printf("Matched %d rows.\n",n-t);
    q=0;
    bool brokethru = false;
    while (1)
    {
      while (q<t)
      {
        // Begin explore node q of the forest 19
        {
          k=unchosen_row[q];
          s=row_dec[k];
          for (l=0;l<n;l++)
            if (slack[l] != 0)
            {
              double del;
              del=costs(k, l)-s+col_inc[l];
              if (del<slack[l])
              {
                if (del == 0)
                {
                  if (row_mate[l]<0){
			brokethru = true;
			break;
		  }
                  slack[l]=0;
                  parent_row[l]=k;
                  //if(n>=24) printf("node %d: row %d==col %d--row %d\n",
                  //   t,row_mate[l],l,k);
                  unchosen_row[t++]=row_mate[l];
                }
                else
                {
                  slack[l]=del;
                  slack_row[l]=k;
                }
              }
            }
	  if(brokethru){
		break;
	  }
        }
        // End explore node q of the forest 19
        q++;
      }
      if(brokethru){
	break;
	}

      // Begin introduce a new zero into the matrix 21
      s=INT_MAX; //INFINITY;
      for (l=0;l<n;l++)
        if (slack[l] != 0 && slack[l]<s)
          s=slack[l];
      for (q=0;q<t;q++)
        row_dec[unchosen_row[q]]+=s;
      for (l=0;l<n;l++)
        if (slack[l] != 0)
        {
          slack[l]-=s;
          if (slack[l] == 0)
          {
            // Begin look at a new zero 22
            k=slack_row[l];
            //if(n>=24) printf("Decreasing uncovered elements by %f produces zero at [%d,%d] rowmate=%d\n",s,k,l,row_mate[l]);
            if (row_mate[l]<0)
            {
              for (j=l+1;j<n;j++)
                if (slack[j] == 0)
                  col_inc[j]+=s;
              brokethru = true;
	      break;
            }
            else
            {
              parent_row[l]=k;
              //if(n>=24) printf("node %d: row %d==col %d--row %d\n",t,row_mate[l],l,k);
              unchosen_row[t++]=row_mate[l];
            }
            // End look at a new zero 22
          }
        }
        else
          col_inc[l]+=s;
      if(brokethru){
	break;
      }
      // End introduce a new zero into the matrix 21
    }
    // Begin update the matching 20
    //if(n>=24) printf("Breakthrough at node %d of %d!\n",q,t);
    while (1)
    {
      j=col_mate[k];
      col_mate[k]=l;
      row_mate[l]=k;
      //if(n>=24) printf("rematching col %d==row %d\n",l,k);
      if(k<0) return INT_MAX; //INFINITY;
      if (j<0)
        break;
      k=parent_row[j];
      l=j;
    }
    // End update the matching 20
    if (--unmatched==0)
      break;
    // Begin get ready for another stage 17
    t=0;
    for (l=0;l<n;l++)
    {
      parent_row[l]= -1;
      slack[l]=INT_MAX; //INFINITY;
    }
    for (k=0;k<n;k++)
      if (col_mate[k]<0)
      {
        //if(n>=24) printf("node %d: unmatched row %d\n",t,k);
        unchosen_row[t++]=k;
      }
    // End get ready for another stage 17
  }
  }

  // Begin doublecheck the solution 23
  for (k=0;k<n;k++)
    for (l=0;l<n;l++)
//      if (costs(k, l)<row_dec[k]-col_inc[l])
      if (costs(k, l)<row_dec[k]-col_inc[l])
      {
        //if(n>=24) printf("problem 1 returning infinity; %f %f %f %f\n",costs(k, l),
        //    row_dec[k], col_inc[l], row_dec[k]-col_inc[l]);
        //for (i=0;i<n;++i)
        //{
        //  if(n>=24) printf("%d ",col_mate[i]);
        //}printf("\n");
        return INT_MAX; //INFINITY;
      }
  for (k=0;k<n;k++)
  {
    l=col_mate[k];
    if (l<0 || costs(k, l) != row_dec[k]-col_inc[l])
    {
      //if(n>=24) printf("problem 2 returning infinity\n");
      return INT_MAX; //INFINITY;
    }
  }
  k=0;
  for (l=0;l<n;l++)
    if (col_inc[l] != 0)
      k++;
  if (k>n)
  {
    //if(n>=24) printf("problem 3 returning infinity\n");
    return INT_MAX; //INFINITY;
  }
  // End doublecheck the solution 23
  // End Hungarian algorithm 18

  matchings = std::vector<int>(n, -1);
  for (i=0;i<n;++i)
  {
    matchings[i] = col_mate[i];
  }
  /*for (k=0;k<n;++k)
  {
    for (l=0;l<n;++l)
    {
      costs(k, l)=costs(k, l)-row_dec[k]+col_inc[l];
    }
   //TRACE("\n");
  }*/
  for (i=0;i<n;i++)
    cost+=row_dec[i];
  for (i=0;i<n;i++)
    cost-=col_inc[i];


  free(slack);
  free(col_inc);
  free(parent_row);
  free(row_mate);
  free(slack_row);
  free(row_dec);
  free(unchosen_row);
  free(col_mate);
  return cost;
}

