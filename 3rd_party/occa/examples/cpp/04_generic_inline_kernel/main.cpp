#include <iostream>

#include <occa.hpp>
#include <occa/experimental.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  occa::setDevice((std::string) args["options/device"]);

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  // Uses the background device
  occa::array<float> array_a(entries);
  occa::array<float> array_b(entries);
  occa::array<float> array_ab(entries);

  // Copy over host data
  array_a.copyFrom(a);
  array_b.copyFrom(b);
  array_ab.fill(0);

  occa::scope scope({
    {"entries", entries},
    {"a", array_a},
    {"b", array_b},
    {"ab", array_ab}
  }, {
    // Define TILE_SIZE at compile-time
    {"defines/TILE_SIZE", 16}
  });

  OCCA_JIT(scope, (
    for (int i = 0; i < entries; ++i; @tile(TILE_SIZE, @outer, @inner)) {
      ab[i] = a[i] + b[i];
    }
  ));

  array_ab.copyTo(ab);

  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

  // Free host memory
  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example showing inline OKL code"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'Serial'}\")")
      .withArg()
      .withDefaultValue("{mode: 'Serial'}")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
