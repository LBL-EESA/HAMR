#include "hamr_buffer.h"

#include <iostream>

// **************************************************************************
template<typename T>
void initialize_cpu(T *data, double val, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        data[i] = val;
    }
}

// **************************************************************************
template <typename T>
hamr::p_buffer<T> initialize_cpu(size_t n_vals, const T &val)
{
    // allocate the memory
    hamr::p_buffer<T> ao = hamr::buffer<T>::New(hamr::buffer<T>::malloc);
    ao->resize(n_vals);

    auto spao = ao->get_cpu_accessible();
    T *pao = spao.get();


    // initialize the data
    initialize_cpu(pao, val, n_vals);

    std::cerr << "initialized to an array of " << n_vals << " to " << val << std::endl;
    if (n_vals < 33)
    {
        std::cerr << "ao = "; ao->print(); std::cerr << std::endl;
        ao->print();
        std::cerr << std::endl;
    }

    return ao;
}






// **************************************************************************
template<typename T, typename U>
void add_cpu(T *result, const T *array_1, const U *array_2, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        result[i] = array_1[i] + array_2[i];
    }
}

// **************************************************************************
template <typename T, typename U>
hamr::p_buffer<T> add_cpu(const hamr::const_p_buffer<T> &a1,
    const hamr::const_p_buffer<U> &a2)
{
    // get the inputs
    auto spa1 = a1->get_cpu_accessible();
    const T *pa1 = spa1.get();

    auto spa2 = a2->get_cpu_accessible();
    const U *pa2 = spa2.get();

    // allocate the memory
    size_t n_vals = a1->size();
    hamr::p_buffer<T> ao = hamr::buffer<T>::New(hamr::buffer<T>::malloc);
    ao->resize(n_vals, T(0));

    auto spao = ao->get_cpu_accessible();
    T *pao = spao.get();

    // initialize the data
    add_cpu(pao, pa1, pa2, n_vals);

    std::cerr << "added " << n_vals << " array " << typeid(T).name()
        << sizeof(T) << " to array  " << typeid(U).name() << sizeof(U) << std::endl;
    if (n_vals < 33)
    {
        std::cerr << "a1 = "; a1->print(); std::cerr << std::endl;
        std::cerr << "a2 = "; a2->print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->print(); std::cerr << std::endl;
    }

    return ao;
}





// **************************************************************************
template<typename T, typename U>
void multiply_scalar_cpu(T *result, const T *array_in, U scalar, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        result[i] = array_in[i] * scalar;
    }
}

// **************************************************************************
template <typename T, typename U>
hamr::p_buffer<T> multiply_scalar_cpu(const hamr::const_p_buffer<T> &ain, const U &val)
{
    // get the inputs
    auto spain = ain->get_cpu_accessible();
    const T *pain = spain.get();

    // allocate the memory
    size_t n_vals = ain->size();
    hamr::p_buffer<T> ao = hamr::buffer<T>::New(hamr::buffer<T>::malloc);
    ao->resize(n_vals, T(0));

    auto spao = ao->get_cpu_accessible();
    T *pao = spao.get();

    // initialize the data
    multiply_scalar_cpu(pao, pain, val, n_vals);

    std::cerr << "multiply_scalar " << val << " " << typeid(U).name() << sizeof(U)
       << " by " << n_vals << " array " << typeid(T).name() << sizeof(T);

    if (n_vals < 33)
    {
        std::cerr << "ain = "; ain->print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->print(); std::cerr << std::endl;
    }

    return ao;
}



// **************************************************************************
template <typename T>
int compare_int(const hamr::const_p_buffer<T> &ain, int val)
{
    size_t n_vals = ain->size();
    std::cerr << "comparing array with " << n_vals << " elements to " << val << std::endl;

    hamr::p_buffer<int> ai = hamr::buffer<int>::New(ain->get_allocator());
    ai->resize(n_vals);
    ain->get(ai);

    if (n_vals < 33)
    {
        ai->print();
    }

    auto spai = ai->get_cpu_accessible();
    int *pai = spai.get();

    for (size_t i = 0; i < n_vals; ++i)
    {
        if (pai[i] != val)
        {
            std::cerr << "ERROR: pai[" << i << "] = " << pai[i] << " != " << val << std::endl;
            return -1;
        }
    }

    std::cerr << "all elements are equal to " << val << std::endl;

    return 0;
}



int main(int, char **)
{
    size_t n_vals = 100000;

    hamr::p_buffer<float>  ao0 = hamr::buffer<float>::New(hamr::buffer<float>::malloc, n_vals, 1.0f);   // = 1 (CPU)
    hamr::p_buffer<float>  ao1 = multiply_scalar_cpu(const_ptr(ao0), 2.0f);                             // = 2 (CPU)
    ao0 = nullptr;

    hamr::p_buffer<double> ao2 = initialize_cpu(n_vals, 2.0);                                           // = 2 (CPU)
    hamr::p_buffer<double> ao3 = add_cpu(const_ptr(ao2), const_ptr(ao1));                               // = 4 (CPU)
    ao1 = nullptr;
    ao2 = nullptr;

    hamr::p_buffer<double> ao4 = multiply_scalar_cpu(const_ptr(ao3), 1000.0);                           // = 4000 (CPU)
    ao3 = nullptr;

    hamr::p_buffer<float>  ao5 = hamr::buffer<float>::New(hamr::buffer<float>::malloc, n_vals, 3.0f);   // = 1 (CPU)
    hamr::p_buffer<float>  ao6 = multiply_scalar_cpu(const_ptr(ao5), 100.0f);                           // = 300 (CPU)
    ao5 = nullptr;

    hamr::p_buffer<float> ao7 = hamr::buffer<float>::New(hamr::buffer<float>::malloc, n_vals);          // = uninit (CPU)
    ao7->set(const_ptr(ao6));                                                                           // = 300 (CPU)
    ao6 = nullptr;

    hamr::p_buffer<double> ao8 = add_cpu(const_ptr(ao4), const_ptr(ao7));                               // = 4300 (CPU)
    ao4 = nullptr;
    ao7 = nullptr;

    return compare_int(const_ptr(ao8), 4300);

    /*
    hamr::p_buffer<double> ao9 = hamr::buffer<double>::New(hamr::buffer<float>::malloc);                // = empty (CPU)
    ao9->assign(const_ptr(ao8));                                                                        // = 4300 (CPU)
    ao8 = nullptr;

    ao9->print();
    ao9 = nullptr;

    return 0;
    */
}

