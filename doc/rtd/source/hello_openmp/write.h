template <typename T>
void write(std::ostream &os, const hamr::buffer<T> &ai)
{
    // get pointer to the input array that is safe to use on the CPU
    auto [spai, pai] = hamr::get_cpu_accessible(ai);

    // write the elements of the array to the stream
    for (size_t i = 0; i < ai.size(); ++i)
    {
        os << pai[i] << " ";
    }

    os << std::endl;
}
