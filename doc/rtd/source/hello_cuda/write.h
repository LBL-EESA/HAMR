template <typename T>
void write(std::ostream &os, const hamr::buffer<T> &ai)
{
    // get pointer to the input array that is safe to use on the host
    auto spai = ai.get_host_accessible();
    const T *pai = spai.get();

    // write the elements of the array to the stream
    for (int i = 0; i < ai.size(); ++i)
    {
        os << pai[i] << " ";
    }

    os << std::endl;
}
