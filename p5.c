
#include<stdio.h>
#include<stdlib.h>
#include<ctype.h>
int top=-1;
char stack[20], post[20];
void push(char item)
{
	stack[++top]=item;
}
int pop()
{
	return stack[top--];
}
int isop(char op)
{
	switch(op)
	{
		case'+':
		case'-': 
		case'*':
		case'/':return 1;
	}
	return 0;
}
int precedence(char op)
{
	switch(op)
	{
		case'(':return 0;
		case'+':
		case'-':return 1;
		case'*':
		case'/':return 2;
	}
}
void intpos()
{
	int i=0,j=0;
	char infix[20],item,in;
	printf("enter the infix expression: ");
	scanf("%s",infix);
	push('#');
	while(infix[i]!='\0')
	{
		in=infix[i];
		if(isalpha(in)||isdigit(in))
		{
			post[j++]=in;
		}
		else if(in=='(')
		{
			push(in);
		}
		else if(in==')')
		{
			while(stack[top]!='(')
			{
				post[j++]=pop();
			}
			item=pop();
		}
		else
		{
			while(precedence(stack[top])>=precedence(in))
			{
				post[j++]=pop();
			}
			push(in);
		}
		i++;
	}
	while(stack[top]!='#')
	{
		post[j++]=pop();
	}
	item=pop();
	top=-1;
	post[j]='\0';
	printf("postfix expression is : %s\n",post);
}
void main()
{
int k=0,i;
	char v1,v2,x,y;
	intpos();
	for(i=0;post[i]!='\0';i++)
	{
		if(isop(post[i]))
		{
			v2=pop();
			v1=pop();
			if(isdigit(v1))
				x='t';
			else 
				x=' ';
			if(isdigit(v2))
				y='t';
			else 
				y=' ';
			printf("t%d=%c%c%c%c%c\n",k,x,v1,post[i],y,v2);
			push(k+48);
			k++;
		}
		else
		{
			push(post[i]);
		}
	
	}
}
