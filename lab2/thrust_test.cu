#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>


using namespace std;



__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct filter_trans{
    __host__ __device__ bool operator()(const char &text){
        return text != '\n';    
    }
};




int main(void)
{



    const char* text="abc sd ff gg d";
    int text_size=15;

    int pos[16]={0};


    for (int i=0;i<10;i++)
        cout<<pos[i]<<" ";
    cout<<endl;
   

    thrust::device_ptr<int> d_pos(pos);
    thrust::device_ptr<const char> d_text(text);
    int *buffer;
	cudaMalloc(&buffer, sizeof(int)*text_size);
    
	thrust::device_ptr<int> d_flag(buffer);
    thrust::transform(d_text, d_text+text_size, d_flag,filter_trans());
    for (int i=1;i<10;i++)
        cout<<(int)*(d_flag+i);

	thrust::inclusive_scan(d_flag, d_flag +text_size, d_pos);

    //for (int i=1;i<10;i++)
      //ut<<buffer[i]<<" ";


}
    //int n = sizeof(text);

    //cout<<n<<endl;
/*

    thrust::device_ptr<char> (test);  
    thrust::device_ptr<int> first = device_ptr;
    thrust::device_ptr<int> last  = device_ptr + 10;
    std::cout << "device array contains " << (last - first) << " values\n";
    thrust::sequence(first, last);
    std::cout << "sum of values is " << thrust::reduce(first, last) << "\n";
  
    //for(size_t i = 0; i < 5; i++)
        std::cout << "d_text[" << i << "] = " << d_text[i] << std::endl;


*/
/*

    // H has storage for 4 integers
    thrust::host_vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;
    
    // H.size() returns the size of vector H
    std::cout << "H has size " << H.size() << std::endl;

    // print contents of H
    for(size_t i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // resize H
    H.resize(2);
    
    std::cout << "H now has size " << H.size() << std::endl;

    // Copy host_vector H to device_vector D
    thrust::device_vector<int> D = H;
    
    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;
    
    // print contents of D
    for(size_t i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // H and D are automatically deleted when the function returns
    return 0;
}


/*

void CountPosition1(const char *text, int *pos, int text_size)
{   
    
    thrust::device_ptr<const char> d_text(text);
    thrust::device_ptr<int> d_pos(pos);
    int *buffer;
    int buffer2[text_size+1]={0};
    cudaMalloc(&buffer, sizeof(int)*text_size);
    //cudaMalloc(&buffer, sizeof(int)*text_size);
    thrust::device_ptr<int> d_flag(buffer);
    //thrust::device_ptr<int> d_zero(buffer2);

    thrust::transform(d_text, d_text+text_size, d_flag,filter_trans());




    //thrust::equal_to<int> binary_pred;
    //thrust::plus<int>     binary_op;
    //thrust::inclusive_scan_by_key(d_flag, d_flag + text_size, buffer2, pos, binary_pred, binary_op); // in-place scan


    cudaFree(buffer);
    //cudaFree(buffer2);

    

}*/