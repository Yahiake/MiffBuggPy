/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "test_utils.hpp"

#include <random>

#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
namespace detail
{
    template <class T_out, class T_in>
    inline std::enable_if_t<std::is_unsigned_v<T_in> && std::is_integral_v<T_out>, bool>
    is_convertible(T_in value)
    {
        return static_cast<uint64_t>(value) <= static_cast<uint64_t>(std::numeric_limits<T_out>::max());
    }

    template <class T_out, class T_in>
    inline std::enable_if_t<std::is_integral_v<T_in> && std::is_signed_v<T_in> && std::is_integral_v<T_out> && std::is_signed_v<T_out>, bool>
    is_convertible(T_in value)
    {
        int64_t signed_value = static_cast<int64_t>(value);
        return signed_value < static_cast<int64_t>(std::numeric_limits<T_out>::max()) && signed_value >= static_cast<int64_t>(std::numeric_limits<T_out>::lowest());
    }

    template <class T_out, class T_in>
    inline std::enable_if_t<std::is_integral_v<T_in> && std::is_signed_v<T_in> && std::is_unsigned_v<T_out>, bool>
    is_convertible(T_in value)
    {
        return value >= 0 && is_convertible<T_out>(static_cast<uint64_t>(value));
    }

    template <class T_out, class T_in>
    inline std::enable_if_t<std::is_floating_point_v<T_in> && std::is_integral_v<T_out>, bool>
    is_convertible(T_in value)
    {
        return value < static_cast<T_in>(std::numeric_limits<T_out>::max()) && value >= static_cast<T_in>(std::numeric_limits<T_out>::lowest());
    }

    template <class T_out, class T_in>
    inline std::enable_if_t<std::is_floating_point_v<T_out>, bool>
    is_convertible(T_in)
    {
        return true;
    }

    template <typename Arch, typename From, typename To>
    constexpr bool uses_fast_cast_v = std::is_same_v<xsimd::kernel::detail::conversion_type<Arch, From, To>,
                                                     xsimd::kernel::detail::with_fast_conversion>;
}

template <class CP>
struct batch_cast_test
{
    static constexpr size_t N = CP::size;
    static constexpr size_t A = CP::alignment;

    using int8_batch = xsimd::batch<int8_t>;
    using uint8_batch = xsimd::batch<uint8_t>;
    using int16_batch = xsimd::batch<int16_t>;
    using uint16_batch = xsimd::batch<uint16_t>;
    using int32_batch = xsimd::batch<int32_t>;
    using uint32_batch = xsimd::batch<uint32_t>;
    using int64_batch = xsimd::batch<int64_t>;
    using uint64_batch = xsimd::batch<uint64_t>;
    using float_batch = xsimd::batch<float>;
    using double_batch = xsimd::batch<double>;

    std::vector<uint64_t> int_test_values;
    std::vector<float> float_test_values;
    std::vector<double> double_test_values;

    batch_cast_test()
    {
        int_test_values = {
            0,
            0x01,
            0x7f,
            0x80,
            0xff,
            0x0100,
            0x7fff,
            0x8000,
            0xffff,
            0x00010000,
            0x7fffffff,
            0x80000000,
            0xffffffff,
            0x0000000100000000,
            0x7fffffffffffffff,
            0x8000000000000000,
            0xffffffffffffffff
        };

        float_test_values = {
            0.0f,
            1.0f,
            -1.0f,
            127.0f,
            128.0f,
            -128.0f,
            255.0f,
            256.0f,
            -256.0f,
            32767.0f,
            32768.0f,
            -32768.0f,
            65535.0f,
            65536.0f,
            -65536.0f,
            2147483647.0f,
            2147483648.0f,
            -2147483648.0f,
            4294967167.0f
        };

        double_test_values = {
            0.0,
            1.0,
            -1.0,
            127.0,
            128.0,
            -128.0,
            255.0,
            256.0,
            -256.0,
            32767.0,
            32768.0,
            -32768.0,
            65535.0,
            65536.0,
            -65536.0,
            2147483647.0,
            2147483648.0,
            -2147483648.0,
            4294967295.0,
            4294967296.0,
            -4294967296.0,
            9223372036854775807.0,
            9223372036854775808.0,
            -9223372036854775808.0,
            18446744073709550591.0
        };
    }

    void test_cast_all_lanes() const
    {
        test_cast_all_lanes_impl<float_batch, int32_batch>("batch cast float -> int32");
        test_cast_all_lanes_impl<double_batch, int64_batch>("batch cast double -> int64");
    }

    void test_bool_cast() const
    {
        test_bool_cast_impl<float_batch, int32_batch>("batch bool cast float -> int32");
        test_bool_cast_impl<float_batch, uint32_batch>("batch bool cast float -> uint32");
        test_bool_cast_impl<int32_batch, float_batch>("batch bool cast int32 -> float");
        test_bool_cast_impl<uint32_batch, float_batch>("batch bool cast uint32 -> float");
        test_bool_cast_impl<float_batch, float_batch>("batch bool cast float -> float");
    }

    void test_cast() const
    {
        for (const auto& test_value : int_test_values)
        {
            test_cast_impl<int8_batch, int8_batch>(test_value, "batch cast int8 -> int8");
            test_cast_impl<int8_batch, uint8_batch>(test_value, "batch cast int8 -> uint8");
            test_cast_impl<uint8_batch, int8_batch>(test_value, "batch cast uint8 -> int8");
            test_cast_impl<uint8_batch, uint8_batch>(test_value, "batch cast uint8 -> uint8");

            test_cast_impl<int16_batch, int16_batch>(test_value, "batch cast int16 -> int16");
            test_cast_impl<int16_batch, uint16_batch>(test_value, "batch cast int16 -> uint16");
            test_cast_impl<uint16_batch, int16_batch>(test_value, "batch cast uint16 -> int16");
            test_cast_impl<uint16_batch, uint16_batch>(test_value, "batch cast uint16 -> uint16");

            test_cast_impl<int32_batch, int32_batch>(test_value, "batch cast int32 -> int32");
            test_cast_impl<int32_batch, uint32_batch>(test_value, "batch cast int32 -> uint32");
            test_cast_impl<int32_batch, float_batch>(test_value, "batch cast int32 -> float");
            test_cast_impl<uint32_batch, int32_batch>(test_value, "batch cast uint32 -> int32");
            test_cast_impl<uint32_batch, uint32_batch>(test_value, "batch cast uint32 -> uint32");
            test_cast_impl<uint32_batch, float_batch>(test_value, "batch cast uint32 -> float");

            test_cast_impl<int64_batch, int64_batch>(test_value, "batch cast int64 -> int64");
            test_cast_impl<int64_batch, uint64_batch>(test_value, "batch cast int64 -> uint64");
            test_cast_impl<int64_batch, double_batch>(test_value, "batch cast int64 -> double");
            test_cast_impl<uint64_batch, int64_batch>(test_value, "batch cast uint64 -> int64");
            test_cast_impl<uint64_batch, uint64_batch>(test_value, "batch cast uint64 -> uint64");
            test_cast_impl<uint64_batch, double_batch>(test_value, "batch cast uint64 -> double");
        }

        for (const auto& test_value : float_test_values)
        {
            test_cast_impl<float_batch, int32_batch>(test_value, "batch cast float -> int32");
            test_cast_impl<float_batch, uint32_batch>(test_value, "batch cast float -> uint32");
            test_cast_impl<float_batch, float_batch>(test_value, "batch cast float -> float");
        }

        for (const auto& test_value : double_test_values)
        {
            test_cast_impl<double_batch, int64_batch>(test_value, "batch cast double -> int64");
            test_cast_impl<double_batch, uint64_batch>(test_value, "batch cast double -> uint64");
            test_cast_impl<double_batch, double_batch>(test_value, "batch cast double -> double");
        }
    }

#if 0 && XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
    template <size_t Align = A>
    std::enable_if_t<Align >:type test_cast_sizeshift1() const
    {
        for (const auto& test_value : int_test_values)
        {
            test_cast_impl<int8_batch, int16_batch>(test_value, "batch cast int8 -> int16");
            test_cast_impl<int8_batch, uint16_batch>(test_value, "batch cast int8 -> uint16");
            test_cast_impl<uint8_batch, int16_batch>(test_value, "batch cast uint8 -> int16");
            test_cast_impl<uint8_batch, uint16_batch>(test_value, "batch cast uint8 -> uint16");

            test_cast_impl<int16_batch, int8_batch>(test_value, "batch cast int16 -> int8");
            test_cast_impl<int16_batch, uint8_batch>(test_value, "batch cast int16 -> uint8");
            test_cast_impl<int16_batch, int32_batch>(test_value, "batch cast int16 -> int32");
            test_cast_impl<int16_batch, uint32_batch>(test_value, "batch cast int16 -> uint32");
            test_cast_impl<int16_batch, float_batch>(test_value, "batch cast int16 -> float");
            test_cast_impl<uint16_batch, int8_batch>(test_value, "batch cast uint16 -> int8");
            test_cast_impl<uint16_batch, uint8_batch>(test_value, "batch cast uint16 -> uint8");
            test_cast_impl<uint16_batch, int32_batch>(test_value, "batch cast uint16 -> int32");
            test_cast_impl<uint16_batch, uint32_batch>(test_value, "batch cast uint16 -> uint32");
            test_cast_impl<uint16_batch, float_batch>(test_value, "batch cast uint16 -> float");

            test_cast_impl<int32_batch, int16_batch>(test_value, "batch cast int32 -> int16");
            test_cast_impl<int32_batch, uint16_batch>(test_value, "batch cast int32 -> uint16");
            test_cast_impl<int32_batch, int64_batch>(test_value, "batch cast int32 -> int64");
            test_cast_impl<int32_batch, uint64_batch>(test_value, "batch cast int32 -> uint64");
            test_cast_impl<int32_batch, double_batch>(test_value, "batch cast int32 -> double");
            test_cast_impl<uint32_batch, int16_batch>(test_value, "batch cast uint32 -> int16");
            test_cast_impl<uint32_batch, uint16_batch>(test_value, "batch cast uint32 -> uint16");
            test_cast_impl<uint32_batch, int64_batch>(test_value, "batch cast uint32 -> int64");
            test_cast_impl<uint32_batch, uint64_batch>(test_value, "batch cast uint32 -> uint64");
            test_cast_impl<uint32_batch, double_batch>(test_value, "batch cast uint32 -> double");

            test_cast_impl<int64_batch, int32_batch>(test_value, "batch cast int64 -> int32");
            test_cast_impl<int64_batch, uint32_batch>(test_value, "batch cast int64 -> uint32");
            test_cast_impl<int64_batch, float_batch>(test_value, "batch cast int64 -> float");
            test_cast_impl<uint64_batch, int32_batch>(test_value, "batch cast uint64 -> int32");
            test_cast_impl<uint64_batch, uint32_batch>(test_value, "batch cast uint64 -> uint32");
            test_cast_impl<uint64_batch, float_batch>(test_value, "batch cast uint64 -> float");
        }

        for (const auto& test_value : float_test_values)
        {
            test_cast_impl<float_batch, int16_batch>(test_value, "batch cast float -> int16");
            test_cast_impl<float_batch, uint16_batch>(test_value, "batch cast float -> uint16");
            test_cast_impl<float_batch, int64_batch>(test_value, "batch cast float -> int64");
            test_cast_impl<float_batch, uint64_batch>(test_value, "batch cast float -> uint64");
            test_cast_impl<float_batch, double_batch>(test_value, "batch cast float -> double");
        }

        for (const auto& test_value : double_test_values)
        {
            test_cast_impl<double_batch, int32_batch>(test_value, "batch cast double -> int32");
            test_cast_impl<double_batch, uint32_batch>(test_value, "batch cast double -> uint32");
            test_cast_impl<double_batch, float_batch>(test_value, "batch cast double -> float");
        }
    }

    template <size_t Align = A>
    std::enable_if_t<Align < 32>::type test_cast_sizeshift1() const
    {
    }
#endif

#if 0 && XSIMD_X86_INSTR_SET > D_X86_AVX512_VERSION
    template <size_t Align = A>
    std::enable_if_t<Align >:type test_cast_sizeshift2() const
    {
        for (const auto& test_value : int_test_values)
        {
            test_cast_impl<int8_batch, int32_batch>(test_value, "batch cast int8 -> int32");
            test_cast_impl<int8_batch, uint32_batch>(test_value, "batch cast int8 -> uint32");
            test_cast_impl<int8_batch, float_batch>(test_value, "batch cast int8 -> float");
            test_cast_impl<uint8_batch, int32_batch>(test_value, "batch cast uint8 -> int32");
            test_cast_impl<uint8_batch, uint32_batch>(test_value, "batch cast uint8 -> uint32");
            test_cast_impl<uint8_batch, float_batch>(test_value, "batch cast uint8 -> float");

            test_cast_impl<int16_batch, int64_batch>(test_value, "batch cast int16 -> int64");
            test_cast_impl<int16_batch, uint64_batch>(test_value, "batch cast int16 -> uint64");
            test_cast_impl<int16_batch, double_batch>(test_value, "batch cast int16 -> double");
            test_cast_impl<uint16_batch, int64_batch>(test_value, "batch cast uint16 -> int64");
            test_cast_impl<uint16_batch, uint64_batch>(test_value, "batch cast uint16 -> uint64");
            test_cast_impl<uint16_batch, double_batch>(test_value, "batch cast uint16 -> double");

            test_cast_impl<int32_batch, int8_batch>(test_value, "batch cast int32 -> int8");
            test_cast_impl<int32_batch, uint8_batch>(test_value, "batch cast int32 -> uint8");
            test_cast_impl<uint32_batch, int8_batch>(test_value, "batch cast uint32 -> int8");
            test_cast_impl<uint32_batch, uint8_batch>(test_value, "batch cast uint32 -> uint8");

            test_cast_impl<int64_batch, int16_batch>(test_value, "batch cast int64 -> int16");
            test_cast_impl<int64_batch, uint16_batch>(test_value, "batch cast int64 -> uint16");
            test_cast_impl<uint64_batch, int16_batch>(test_value, "batch cast uint64 -> int16");
            test_cast_impl<uint64_batch, uint16_batch>(test_value, "batch cast uint64 -> uint16");
        }

        for (const auto& test_value : float_test_values)
        {
            test_cast_impl<float_batch, int8_batch>(test_value, "batch cast float -> int8");
            test_cast_impl<float_batch, uint8_batch>(test_value, "batch cast float -> uint8");
        }

        for (const auto& test_value : double_test_values)
        {
            test_cast_impl<double_batch, int16_batch>(test_value, "batch cast double -> int16");
            test_cast_impl<double_batch, uint16_batch>(test_value, "batch cast double -> uint16");
        }
    }

    template <size_t Align = A>
    std::enable_if_t<Align < 64>::type test_cast_sizeshift2() const
    {
    }
#endif

private:
    template <class B_in, class B_out, class T>
    void test_cast_impl(T test_value, const std::string& name) const
    {
        using T_in = typename B_in::value_type;
        using T_out = typename B_out::value_type;
        using B_common_in = xsimd::batch<T_in>;
        using B_common_out = xsimd::batch<T_out>;

        auto clamp = [](T v)
        {
            return static_cast<T_in>(xsimd::min(v, static_cast<T>(std::numeric_limits<T_in>::max() - 1)));
        };

        T_in in_test_value = clamp(test_value);
        if (detail::is_convertible<T_out>(in_test_value))
        {
            B_common_out res = xsimd::batch_cast<T_out>(B_common_in(in_test_value));
            INFO(name);
            T_out scalar_ref = static_cast<T_out>(in_test_value);
            T_out scalar_res = res.get(0);
            CHECK_SCALAR_EQ(scalar_ref, scalar_res);
            CHECK_SCALAR_EQ(scalar_ref, xsimd::batch_cast<T_out>(in_test_value));
        }
    }

    // A float -> same-width-int cast can recombine per-lane data, so every lane has to
    // carry a different value; the other cast tests are splats and only look at lane 0.
    template <class B_in, class B_out>
    void test_cast_all_lanes_impl(const std::string& name) const
    {
        using T_in = typename B_in::value_type;
        using T_out = typename B_out::value_type;
        constexpr int digits = std::numeric_limits<T_out>::digits; // 31 or 63
        const T_in beyond = std::ldexp(T_in(1), digits); // one past the top of T_out
        const T_in top = std::nextafter(beyond, T_in(0)); // the largest T_in that fits
        // The ends of the range, then every power of two the cast can reach probed on
        // both sides: those are where a carry between halves and the end of the
        // mantissa live, and a random draw never lands on one of them.
        std::vector<T_in> values = { T_in(0), -T_in(0), -beyond, top, -top };
        for (int e = 0; e < digits; ++e)
        {
            const T_in p = std::ldexp(T_in(1), e);
            for (const T_in v : { p - T_in(1.5), p - T_in(1), p - T_in(.5), std::nextafter(p, T_in(0)),
                                  p, std::nextafter(p, beyond), p + T_in(.5), p + T_in(1) })
            {
                values.push_back(v);
                values.push_back(-v);
            }
        }
        // Eight random values in every binade the cast can reach, both signs.  The
        // engine is default seeded, so a failure reproduces.
        std::default_random_engine generator;
        std::uniform_real_distribution<T_in> mantissa(T_in(1), T_in(2));
        for (int e = -1; e < digits; ++e)
            for (int k = 0; k < 8; ++k)
            {
                const T_in v = std::ldexp(mantissa(generator), e);
                values.push_back(v);
                values.push_back(-v);
            }
        constexpr size_t n = B_in::size;
        T_in buffer[n];
        for (size_t i = 0; i < values.size(); i += n)
        {
            for (size_t l = 0; l < n; ++l)
                buffer[l] = values[(i + l) % values.size()];
            B_out res = xsimd::batch_cast<T_out>(B_in::load_unaligned(buffer));
            for (size_t l = 0; l < n; ++l)
            {
                INFO(name, ", lane ", l, " holding ", buffer[l]);
                CHECK_SCALAR_EQ(static_cast<T_out>(buffer[l]), res.get(l));
            }
        }
    }

    template <class B_in, class B_out>
    void test_bool_cast_impl(const std::string& name) const
    {
        using T_in = typename B_in::value_type;
        using T_out = typename B_out::value_type;
        using B_common_in = xsimd::batch_bool<T_in>;
        using B_common_out = xsimd::batch_bool<T_out>;

        B_common_in all_true_in(true);
        B_common_out all_true_res = xsimd::batch_bool_cast<T_out>(all_true_in);
        INFO(name);
        CHECK_SCALAR_EQ(all_true_res.get(0), true);
        CHECK_SCALAR_EQ(xsimd::batch_bool_cast<B_out>(true), true);

        B_common_in all_false_in(false);
        B_common_out all_false_res = xsimd::batch_bool_cast<T_out>(all_false_in);
        INFO(name);
        CHECK_SCALAR_EQ(all_false_res.get(0), false);
        CHECK_SCALAR_EQ(xsimd::batch_bool_cast<B_out>(false), false);
    }
};

TEST_CASE_TEMPLATE("[xsimd cast tests]", B, CONVERSION_TYPES)
{
    batch_cast_test<B> Test;

    SUBCASE("bool cast")
    {
        Test.test_bool_cast();
    }

    SUBCASE("cast")
    {
        Test.test_cast();
    }

    SUBCASE("cast all lanes")
    {
        Test.test_cast_all_lanes();
    }
}
#endif
#if 0 && XSIMD_X86_INSTR_SET > D_X86_AVX_VERSION
TYPED_TEST(batch_cast_test, cast_sizeshift1)
{
    this->test_cast_sizeshift1();
}
#endif

#if 0 && XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
TYPED_TEST(batch_cast_test, cast_sizeshift2)
{
    this->test_cast_sizeshift2();
}
#endif

// neon32 has no batch<double>, and detail::uses_fast_cast_v lives inside the guard
// that excludes it, so the whole block needs the same guard.
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
// sve and rvv used to declare fast_cast in detail_sve / detail_rvv, where the
// dispatcher in kernel::detail cannot see it, so every conversion silently fell back
// to the scalar loop.  int32 <-> float has no common overload, so both asserts below
// fail if an architecture declares its fast_cast outside kernel::detail again.
TEST_CASE_TEMPLATE("[xsimd cast tests]", B, CONVERSION_TYPES)
{
    SUBCASE("use fastcast")
    {
        using A = xsimd::default_arch;
#if XSIMD_WITH_SSE2 || XSIMD_WITH_SVE || XSIMD_WITH_RVV
        static_assert(detail::uses_fast_cast_v<A, int32_t, float>,
                      "expected int32 to float conversion to use fast_cast");
        static_assert(detail::uses_fast_cast_v<A, float, int32_t>,
                      "expected float to int32 conversion to use fast_cast");
#endif
        // the common overload answers double -> int64_t on every architecture
        static_assert(detail::uses_fast_cast_v<A, double, int64_t>,
                      "expected double to int64 conversion to use fast_cast");
    }
}
#endif
#endif
