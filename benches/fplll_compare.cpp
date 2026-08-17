// In-process fplll side of the lattica comparison benchmark.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <fplll/fplll.h>

using fplll::closest_vector;
using fplll::CVPM_FAST;
using fplll::CVPM_PROVED;
using fplll::FT_DEFAULT;
using fplll::LLL_DEF_ETA;
using fplll::LLL_DEFAULT;
using fplll::lll_reduction;
using fplll::LM_WRAPPER;
using fplll::RED_SUCCESS;
using fplll::Z_NR;
using fplll::ZZ_mat;

namespace {

constexpr int dimensions[] = {8, 16, 24};
constexpr std::size_t lll_cases = 16;
constexpr std::size_t targets_per_dimension = 128;
constexpr std::size_t samples = 11;
constexpr std::int64_t cvp_scale = 1'009;
volatile int status_sink = 0;

struct Rng {
  std::uint64_t state;

  std::uint64_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }

  std::size_t index(std::size_t bound) {
    return static_cast<std::size_t>(next() % bound);
  }

  std::int64_t signed_value(std::int64_t bound) {
    const auto width = static_cast<std::uint64_t>(2 * bound + 1);
    return static_cast<std::int64_t>(next() % width) - bound;
  }
};

using Rows = std::vector<std::int64_t>;

Rows canonical_basis(std::size_t dimension) {
  Rows basis(dimension * dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    basis[row * dimension + row] = 2;
    if (row > 0) {
      basis[row * dimension + row - 1] = 1;
    }
    if (row > 1) {
      basis[row * dimension + row - 2] = -1;
    }
  }
  return basis;
}

Rows cvp_basis(std::size_t dimension) {
  Rows basis(dimension * dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    basis[row * dimension + row] = 2;
    if (row + 1 < dimension) {
      basis[row * dimension + row + 1] = 1;
    }
  }
  return basis;
}

Rows skew_basis(std::size_t dimension, std::size_t benchmark_case) {
  auto basis = canonical_basis(dimension);
  Rng rng{0xd1b54a32d192ed03ULL ^ static_cast<std::uint64_t>(dimension) ^
          (static_cast<std::uint64_t>(benchmark_case) << 32)};
  for (std::size_t step = 0; step < 2 * dimension; ++step) {
    const auto destination = rng.index(dimension);
    auto source = rng.index(dimension - 1);
    if (source >= destination) {
      ++source;
    }
    const std::int64_t sign = (rng.next() & 1) == 0 ? -1 : 1;
    const auto destination_start = destination * dimension;
    const auto source_start = source * dimension;
    bool acceptable = true;
    for (std::size_t column = 0; column < dimension; ++column) {
      const auto candidate = basis[destination_start + column] +
                             sign * basis[source_start + column];
      acceptable = acceptable && std::abs(candidate) <= 256;
    }
    if (acceptable) {
      for (std::size_t column = 0; column < dimension; ++column) {
        basis[destination_start + column] +=
            sign * basis[source_start + column];
      }
    }
  }
  return basis;
}

ZZ_mat<mpz_t> to_matrix(const Rows &rows, std::size_t dimension) {
  ZZ_mat<mpz_t> matrix(static_cast<int>(dimension),
                       static_cast<int>(dimension));
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column < dimension; ++column) {
      matrix[static_cast<int>(row)][static_cast<int>(column)] =
          rows[row * dimension + column];
    }
  }
  return matrix;
}

std::vector<Rows> make_targets(const Rows &basis, std::size_t dimension) {
  Rng rng{0xa0761d6478bd642fULL ^ static_cast<std::uint64_t>(dimension)};
  std::vector<Rows> targets;
  targets.reserve(targets_per_dimension);
  for (std::size_t target_index = 0; target_index < targets_per_dimension;
       ++target_index) {
    std::vector<std::int64_t> coordinates(dimension);
    for (auto &coordinate : coordinates) {
      coordinate = rng.signed_value(8);
    }
    Rows target(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
      for (std::size_t column = 0; column < dimension; ++column) {
        target[column] +=
            coordinates[row] * basis[row * dimension + column] * cvp_scale;
      }
    }
    for (auto &entry : target) {
      entry += rng.signed_value(1'000);
    }
    targets.push_back(std::move(target));
  }
  return targets;
}

std::vector<Z_NR<mpz_t>> to_vector(const Rows &values) {
  std::vector<Z_NR<mpz_t>> result(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    result[index] = values[index];
  }
  return result;
}

int reduce(ZZ_mat<mpz_t> &basis) {
  return lll_reduction(basis, 0.99, LLL_DEF_ETA, LM_WRAPPER, FT_DEFAULT, 0,
                       LLL_DEFAULT);
}

double median_ns(std::vector<std::chrono::nanoseconds> timings,
                 std::size_t operations) {
  std::sort(timings.begin(), timings.end());
  return static_cast<double>(timings[timings.size() / 2].count()) /
         static_cast<double>(operations);
}

std::pair<double, std::int64_t> benchmark_lll(std::size_t dimension) {
  std::vector<Rows> rows;
  std::vector<ZZ_mat<mpz_t>> matrices;
  rows.reserve(lll_cases);
  matrices.reserve(lll_cases);
  std::int64_t checksum = 0;
  std::size_t flat_index = 0;
  for (std::size_t benchmark_case = 0; benchmark_case < lll_cases;
       ++benchmark_case) {
    rows.push_back(skew_basis(dimension, benchmark_case));
    for (const auto entry : rows.back()) {
      ++flat_index;
      checksum += static_cast<std::int64_t>(flat_index) * entry;
    }
    matrices.push_back(to_matrix(rows.back(), dimension));
  }

  for (const auto &source : matrices) {
    auto basis = source;
    const int status = reduce(basis);
    if (status != RED_SUCCESS) {
      throw std::runtime_error("fplll LLL warm-up failed");
    }
    status_sink = status;
  }

  std::vector<std::chrono::nanoseconds> timings;
  timings.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (const auto &source : matrices) {
      auto basis = source;
      const int status = reduce(basis);
      if (status != RED_SUCCESS) {
        throw std::runtime_error("fplll LLL failed");
      }
      status_sink = status;
    }
    timings.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start));
  }
  return {median_ns(std::move(timings), lll_cases), checksum};
}

std::tuple<std::int64_t, std::int64_t, std::int64_t>
fingerprints(const ZZ_mat<mpz_t> &basis,
             const std::vector<std::vector<Z_NR<mpz_t>>> &targets,
             const std::vector<std::vector<Z_NR<mpz_t>>> &coordinates) {
  const auto dimension = static_cast<std::size_t>(basis.get_cols());
  std::int64_t target_checksum = 0;
  std::int64_t point_checksum = 0;
  std::int64_t distance_checksum = 0;
  for (std::size_t target_index = 0; target_index < coordinates.size();
       ++target_index) {
    std::int64_t distance = 0;
    for (std::size_t column = 0; column < dimension; ++column) {
      Z_NR<mpz_t> ambient;
      for (std::size_t row = 0; row < dimension; ++row) {
        ambient.addmul(coordinates[target_index][row],
                       basis[static_cast<int>(row)][static_cast<int>(column)]);
      }
      const auto ambient_scaled = ambient.get_si();
      const auto weight =
          static_cast<std::int64_t>(1 + target_index * dimension + column);
      point_checksum += weight * (ambient_scaled / cvp_scale);
      target_checksum += weight * targets[target_index][column].get_si();
      const auto residual =
          targets[target_index][column].get_si() - ambient_scaled;
      distance += residual * residual;
    }
    distance_checksum += static_cast<std::int64_t>(target_index + 1) * distance;
  }
  return {target_checksum, point_checksum, distance_checksum};
}

std::tuple<double, std::int64_t, std::int64_t, std::int64_t>
benchmark_cvp(std::size_t dimension, int method) {
  const auto rows = cvp_basis(dimension);
  auto scaled_rows = rows;
  for (auto &entry : scaled_rows) {
    entry *= cvp_scale;
  }
  auto basis = to_matrix(scaled_rows, dimension);
  if (reduce(basis) != RED_SUCCESS) {
    throw std::runtime_error("fplll CVP preparation failed");
  }
  const auto integer_targets = make_targets(rows, dimension);
  std::vector<std::vector<Z_NR<mpz_t>>> targets;
  targets.reserve(targets_per_dimension);
  for (const auto &target : integer_targets) {
    targets.push_back(to_vector(target));
  }

  std::vector<std::vector<Z_NR<mpz_t>>> outputs(targets_per_dimension);
  for (std::size_t index = 0; index < targets.size(); ++index) {
    const int status =
        closest_vector(basis, targets[index], outputs[index], method);
    if (status != RED_SUCCESS) {
      throw std::runtime_error("fplll CVP warm-up failed");
    }
  }
  const auto [target_checksum, point_checksum, distance_checksum] =
      fingerprints(basis, targets, outputs);

  std::vector<std::chrono::nanoseconds> timings;
  timings.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < targets.size(); ++index) {
      const int status =
          closest_vector(basis, targets[index], outputs[index], method);
      if (status != RED_SUCCESS) {
        throw std::runtime_error("fplll CVP failed");
      }
      status_sink = status;
    }
    timings.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start));
  }
  return {median_ns(std::move(timings), targets_per_dimension), target_checksum,
          point_checksum, distance_checksum};
}

} // namespace

int main() {
  try {
    std::cout << "library,operation,dimension,median_ns,target_fingerprint,"
                 "point_fingerprint,distance_fingerprint\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const int dimension : dimensions) {
      const auto [lll_ns, basis_checksum] =
          benchmark_lll(static_cast<std::size_t>(dimension));
      std::cout << "fplll,lll," << dimension << ',' << lll_ns << ','
                << basis_checksum << ",0,0\n";
      const auto [fast_ns, fast_target, fast_point, fast_distance] =
          benchmark_cvp(static_cast<std::size_t>(dimension), CVPM_FAST);
      std::cout << "fplll,cvp_public_fast," << dimension << ',' << fast_ns
                << ',' << fast_target << ',' << fast_point << ','
                << fast_distance << '\n';
      const auto [proved_ns, proved_target, proved_point, proved_distance] =
          benchmark_cvp(static_cast<std::size_t>(dimension), CVPM_PROVED);
      std::cout << "fplll,cvp_public_proved," << dimension << ',' << proved_ns
                << ',' << proved_target << ',' << proved_point << ','
                << proved_distance << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return status_sink;
}
