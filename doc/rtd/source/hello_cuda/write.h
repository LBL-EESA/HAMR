#ifndef write_h
#define write_h

template <typename T>
void write(std::ostream &os, const hamr::buffer<T> &ai)
{
    // get pointer to the input array that is safe to use on the CPU
    auto spai = ai.get_cpu_accessible();
    const T *pai = spai.get();

    for (int i = 0; i < ai.size(); ++i)
    {
        std::cerr << pai[i] << " ";
    }

    std::cerr << std::endl;
}

#endif
