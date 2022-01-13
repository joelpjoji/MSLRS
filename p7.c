#include<stdio.h>
#include<string.h>
#include<math.h>
int i,j,k,n;
char tac[10][10],quad[10][10],op;
void main()
{
    printf("enter a sequence  ");
    scanf("%d ",&n);
   
        for(j=0;j<n;j++)
            scanf("%s",tac[j]);
       
   
    printf("\n ");
    for(i=0;i<n;i++)
    {   
       
            quad[i][1]=tac[i][2];
        if(tac[i][3]!= NULL)       
            quad[i][0]=tac[i][3];
        else
            quad[i][0]='=';
       
        if(tac[i][4]!= NULL)       
            quad[i][2]=tac[i][4];
        else
            quad[i][2]='-';   

        quad[i][3]=tac[i][0];
    }
    for(i=0;i<n;i++)
    {
        op=quad[i][0];
        printf("MOV R0,%c \n",quad[i][1]);
        switch(op)
        {
            case '+': printf("ADD R0,%c \n",quad[i][2]);
            		break;
            case '-': printf("SUB R0,%c \n",quad[i][2]);
           		 break;
            case '*': printf("MUL R0,%c \n",quad[i][2]);
            		break;
	    case '/': printf("DIV R0,%c \n",quad[i][2]);
            		break;
        }
        printf("MOV %c,R0 \n",quad[i][3]);
    }
}
