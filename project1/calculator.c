#include <math.h>
#include <regex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #define GMP
#ifdef GMP
#include <gmp.h>
#include <mpfr.h>
#endif

// #define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT(x, ...) printf(x, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(x, ...) \
    do                      \
    {                       \
    } while (0)
#endif

// Arguments
#define MAX_ELEMENTS 100
#define MAX_INPUT_LENGTH 100
#define BASE 10
#define PRECISION 300

// Constants
#define NUMBER 1
#define OPERATOR 2
#define LEFT_BRACKET 3
#define RIGHT_BRACKET 4

typedef struct Element
{
    int type;
// mpf_t number;
#ifdef GMP
    mpf_t number;
#else
    double number;
#endif
    char op;
    int opPriority;
    struct Element *next;
} Element;

/*
Stack implementation
 */

typedef struct
{
    Element *top;
    int size;
} Stack;

void initStack(Stack *stack)
{
    stack->top = NULL;
    stack->size = 0;
}

Element *pop(Stack *stack)
{
    if (stack->top == NULL)
    {
        return NULL;
    }
    Element *temp = stack->top;
    stack->top = stack->top->next;
    stack->size--;
    return temp;
}

void push(Stack *stack, Element *element)
{
    element->next = stack->top;
    stack->top = element;
    stack->size++;
}

Element *peek(Stack *stack) { return stack->top; }

bool isEmpty(Stack *stack) { return stack->top == NULL; }

/*
Big number implementation
 */
typedef struct
{
    int sign;
    char *number;
} BigNumber;
void add(BigNumber *num1, BigNumber *num2, BigNumber *result)
{
    // Add two big numbers
    int len1 = strlen(num1->number);
    int len2 = strlen(num2->number);
    int carry = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int sum = 0;
    int maxLen = len1 > len2 ? len1 : len2;
    char *temp = (char *)malloc(maxLen + 2);
    for (i = len1 - 1, j = len2 - 1, k = maxLen; i >= 0 && j >= 0; i--, j--, k--)
    {
        sum = (num1->number[i] - '0') + (num2->number[j] - '0') + carry;
        temp[k] = (sum % 10) + '0';
        carry = sum / 10;
    }
    while (i >= 0)
    {
        sum = (num1->number[i] - '0') + carry;
        temp[k] = (sum % 10) + '0';
        carry = sum / 10;
        i--;
        k--;
    }
    while (j >= 0)
    {
        sum = (num2->number[j] - '0') + carry;
        temp[k] = (sum % 10) + '0';
        carry = sum / 10;
        j--;
        k--;
    }
    if (carry > 0)
    {
        temp[k] = carry + '0';
        k--;
    }
    result->number = (char *)malloc(maxLen + 2 - k);
    strcpy(result->number, temp + k + 1);
    free(temp);
}
void sub(BigNumber *num1, BigNumber *num2, BigNumber *result)
{
    // Subtract two big numbers
    int len1 = strlen(num1->number);
    int len2 = strlen(num2->number);
    int borrow = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int diff = 0;
    int maxLen = len1 > len2 ? len1 : len2;
    char *temp = (char *)malloc(maxLen + 2);
    for (i = len1 - 1, j = len2 - 1, k = maxLen; i >= 0 && j >= 0; i--, j--, k--)
    {
        diff = (num1->number[i] - '0') - (num2->number[j] - '0') - borrow;
        if (diff < 0)
        {
            diff += 10;
            borrow = 1;
        }
        else
        {
            borrow = 0;
        }
        temp[k] = diff + '0';
    }
    while (i >= 0)
    {
        diff = (num1->number[i] - '0') - borrow;
        if (diff < 0)
        {
            diff += 10;
            borrow = 1;
        }
        else
        {
            borrow = 0;
        }
        temp[k] = diff + '0';
        i--;
        k--;
    }
    result->number = (char *)malloc(maxLen + 2 - k);
    strcpy(result->number, temp + k + 1);
    free(temp);
}
/*
Calculator implementation
 */

int performOperation(Stack *numStack, Stack *opStack)
{
    Element *op = pop(opStack);
    Element *num1 = pop(numStack);
    Element *num2 = peek(numStack);
#ifdef GMP
    mpfr_t tempExp;
    mpfr_t tempBase;

    switch (op->op)
    {
    case '+':
        mpf_add(num2->number, num2->number, num1->number);
        break;
    case '-':
        mpf_sub(num2->number, num2->number, num1->number);
        break;
    case '*':
        mpf_mul(num2->number, num2->number, num1->number);
        break;
    case '/':
        if (mpf_cmp_ui(num1->number, 0) == 0)
        {
            printf("A number cannot be divied by zero.\n");
            return 1;
        }
        mpf_div(num2->number, num2->number, num1->number);
        break;
    case '^':
        if (mpf_integer_p(num1->number))
        {
            mpf_pow_ui(num2->number, num2->number, mpf_get_ui(num1->number));
        }
        else
        {
            mpfr_init2(tempExp, PRECISION);
            mpfr_init2(tempBase, PRECISION);
            mpfr_set_f(tempExp, num1->number, MPFR_RNDN);
            mpfr_set_f(tempBase, num2->number, MPFR_RNDN);
            mpfr_pow(tempBase, tempBase, tempExp, MPFR_RNDN);
            mpfr_get_f(num2->number, tempBase, MPFR_RNDN);
            mpfr_clear(tempExp);
            mpfr_clear(tempBase);
        }
        break;
    default:
        break;
    }
#else
    switch (op->op)
    {
    case '+':
        num2->number += num1->number;
        break;
    case '-':

        num2->number -= num1->number;
        break;
    case '*':
        num2->number *= num1->number;
        break;
    case '/':
        if (num1->number == 0)
        {
            printf("A number cannot be divied by zero.\n");
            return 1;
        }
        num2->number /= num1->number;
        break;
    case '^':
        num2->number = pow(num2->number, num1->number);
        break;
    default:
        break;
    }
#endif
    return 0;
}

void printResult(Stack *numStack)
{
#ifdef GMP
    if (mpf_cmp_d(peek(numStack)->number, 1e10) > 0 || mpf_cmp_d(peek(numStack)->number, -1e10) < 0)
    {
        mpf_out_str(stdout, BASE, PRECISION, peek(numStack)->number);
        printf("\n");
        // gmp_printf("%.*Ff\n", PRECISION, peek(numStack)->number);
    }
#else
    if (peek(numStack)->number > 1e10 || peek(numStack)->number < -1e10)
    {
        printf("%.*e\n", PRECISION, peek(numStack)->number);
    }
#endif
    else
    {
        char result[PRECISION * 2];
#ifdef GMP
        gmp_sprintf(result, "%Ff", peek(numStack)->number);
#else
        sprintf(result, "%.*f", PRECISION, peek(numStack)->number);
#endif
        unsigned long j = strlen(result) - 1;
        while (result[j] == '0')
        {
            j--;
        }
        if (result[j] == '.')
        {
            j--;
        }
        result[j + 1] = '\0';
        printf("%s\n", result);
    }
}

void calculate(Element *elements)
{
    Stack numStack;
    Stack opStack;
    initStack(&numStack);
    initStack(&opStack);
    int index = 0;
    while (elements[index].type != 0)
    {
        if (elements[index].type == NUMBER)
        {
            push(&numStack, &elements[index]);
        }
        else if (elements[index].type == OPERATOR)
        {
            if (isEmpty(&opStack) ||
                elements[index].opPriority > peek(&opStack)->opPriority ||
                peek(&opStack)->type == LEFT_BRACKET)
            {
                push(&opStack, &elements[index]);
            }
            else
            {
                while (!isEmpty(&opStack) && peek(&opStack)->type == OPERATOR &&
                       elements[index].opPriority <= peek(&opStack)->opPriority)
                {
                    if (performOperation(&numStack, &opStack))
                    {
                        return;
                    }
                }
                push(&opStack, &elements[index]);
            }
        }
        else if (elements[index].type == LEFT_BRACKET)
        {
            push(&opStack, &elements[index]);
        }
        else if (elements[index].type == RIGHT_BRACKET)
        {
            while (peek(&opStack)->type != LEFT_BRACKET)
            {
                if (performOperation(&numStack, &opStack))
                {
                    return;
                }
            }
            pop(&opStack);
        }
        index++;
    }
    while (!isEmpty(&opStack))
    {
        if (performOperation(&numStack, &opStack))
        {
            return;
        }
    }
    printResult(&numStack);
}

regex_t numRegex;
regex_t opRegex;

void removeSpace(char *input)
{
    int i = 0;
    int j = 0;
    while (input[i] != '\0')
    {
        if (input[i] != ' ')
        {
            input[j] = input[i];
            j++;
        }
        i++;
    }
    // remove \n
    if (input[j - 1] == '\n')
    {
        j--;
    }
    input[j] = '\0';
}

int opPriority(char op)
{
    switch (op)
    {
    case '+':
    case '-':
        return 1;
    case '*':
    case '/':
        return 2;
    case '^':
        return 3;
    default:
        return 0;
    }
    return 0;
}

int readExpression(char *input, Element *elements)
{
    regmatch_t pmatch[1];
    int offset = 0;
    int index = 0;
    char buffer[MAX_INPUT_LENGTH];
    bool prevIsOp = true;
    int bracketCount = 0;
    removeSpace(input);
    do
    {
        if (prevIsOp && !regexec(&numRegex, input + offset, 1, pmatch, 0) &&
            pmatch[0].rm_so == 0)
        {
            DEBUG_PRINT("Match: %.*s\n", pmatch[0].rm_eo - pmatch[0].rm_so,
                        input + offset + pmatch[0].rm_so);
            elements[index].type = NUMBER;
#ifdef GMP
            snprintf(buffer, pmatch[0].rm_eo - pmatch[0].rm_so + 1, "%s",
                     input + offset + pmatch[0].rm_so);
            mpf_init2(elements[index].number, PRECISION);
            mpf_set_str(elements[index].number, buffer, BASE);
#else
            elements[index].number = strtof(input + offset + pmatch[0].rm_so, NULL);
#endif
            offset += pmatch[0].rm_eo;
            prevIsOp = false;
        }
        else if (!prevIsOp && !regexec(&opRegex, input + offset, 1, pmatch, 0) &&
                 pmatch[0].rm_so == 0)
        {
            DEBUG_PRINT("Match: %.*s\n", pmatch[0].rm_eo - pmatch[0].rm_so,
                        input + offset + pmatch[0].rm_so);
            elements[index].type = OPERATOR;
            elements[index].op = input[offset];
            elements[index].opPriority = opPriority(input[offset]);
            offset += pmatch[0].rm_eo;
            prevIsOp = true;
        }
        else if (input[offset] == '(' && prevIsOp)
        {
            DEBUG_PRINT("Match: (\n");
            elements[index].type = LEFT_BRACKET;
            offset++;
            bracketCount++;
        }
        else if (input[offset] == ')' && !prevIsOp)
        {
            DEBUG_PRINT("Match: )\n");
            elements[index].type = RIGHT_BRACKET;
            offset++;
            bracketCount--;
        }
        else
        {
            printf("The input cannot be interpret as an expression.\n");
            return 1;
        }
        index++;
    } while (input[offset] != '\0');
    if (bracketCount != 0 || prevIsOp)
    {
        printf("The input cannot be interpret as an expression.\n");
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    regcomp(&numRegex, "-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?", REG_EXTENDED);
    regcomp(&opRegex, "[-\\+\\/*^]", REG_EXTENDED);
    Element elements[MAX_ELEMENTS];
    memset(elements, 0, sizeof(elements));
    char input[MAX_INPUT_LENGTH];
    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            strcat(input, argv[i]);
        }
        if (readExpression(input, elements))
        {
            return 0;
        }
        calculate(elements);
    }
    else if (argc < 2)
    {
        while (1)
        {
            printf(">> ");
            (void)fgets(input, MAX_INPUT_LENGTH, stdin);
            if (strcmp(input, "quit\n") == 0)
            {
                break;
            }
            if (readExpression(input, elements))
            {
                continue;
            }
            calculate(elements);
            memset(elements, 0, sizeof(elements));
        }
    }
    regfree(&numRegex);
    regfree(&opRegex);
    return 0;
}