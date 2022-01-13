
#include<stdio.h>
#include<stdlib.h>
#include<ctype.h>
char a[15];
int i = 0;
void E_ ();
void T () 
{
  
if (isalpha (a[i])) {
    printf ("\nT->id");
    i++;
    E_ ();
}
else {
    printf ("\nerror");
    }
}
void E_ () 
{
    if (a[i] == '+')
    {
        printf ("\nE'-> +TE'");
        i++;
        T ();
    }
  else
    {
      printf ("\nE'->e");
    }
}
void E () {
    printf ("\nE->TE'");
    T ();
} 
void main () 
{
    printf ("Enter the string\n");
    scanf ("%s", a);
    E ();
    if (a[i] == '\0') {
        printf ("\nString is accepted");
        }
    else {
        printf ("\nString not accepted");
        }
}


