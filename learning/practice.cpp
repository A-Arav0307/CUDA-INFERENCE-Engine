#include <iostream> 

int addNumbers(int first_number, int second_number){
    int sum = first_number + second_number; 
    return sum; 
}

int main(){
    int first_number = 13;
    int second_number = 7;
    std::cout << "Sum is " << addNumbers(first_number, second_number) << std::endl;
}