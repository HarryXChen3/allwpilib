// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include <stdint.h>

#include <bitset>
#include <cmath>
#include <span>

#include <gtest/gtest.h>
#include <units/time.h>
#include <units/voltage.h>

#include "sysid/analysis/AnalysisManager.h"
#include "sysid/analysis/AnalysisType.h"
#include "sysid/analysis/ArmSim.h"
#include "sysid/analysis/ElevatorSim.h"
#include "sysid/analysis/FeedforwardAnalysis.h"
#include "sysid/analysis/SimpleMotorSim.h"

namespace {

enum Movements : uint32_t {
  kSlowForward,
  kSlowBackward,
  kFastForward,
  kFastBackward
};

inline constexpr int kMovementCombinations = 16;

/**
 * Return simulated test data for a given simulation model.
 *
 * @tparam Model The model type.
 * @param model The simulation model.
 * @param movements Which movements to do.
 */
template <typename Model>
sysid::Storage CollectData(Model& model, std::bitset<4> movements) {
  constexpr auto kUstep = 0.25_V / 1_s;
  constexpr units::volt_t kUmax = 7_V;
  constexpr units::second_t T = 5_ms;
  constexpr units::second_t kTestDuration = 5_s;

  sysid::Storage storage;
  auto& [slowForward, slowBackward, fastForward, fastBackward] = storage;
  auto voltage = 0_V;

  // Slow forward
  if (movements.test(Movements::kSlowForward)) {
    model.Reset();
    voltage = 0_V;
    for (int i = 0; i < (kTestDuration / T).value(); ++i) {
      slowForward.emplace_back(sysid::PreparedData{
          i * T, voltage.value(), model.GetPosition(), model.GetVelocity(), T,
          model.GetAcceleration(voltage), std::cos(model.GetPosition()),
          std::sin(model.GetPosition())});

      model.Update(voltage, T);
      voltage += kUstep * T;
    }
  }

  // Slow backward
  if (movements.test(Movements::kSlowBackward)) {
    model.Reset();
    voltage = 0_V;
    for (int i = 0; i < (kTestDuration / T).value(); ++i) {
      slowBackward.emplace_back(sysid::PreparedData{
          i * T, voltage.value(), model.GetPosition(), model.GetVelocity(), T,
          model.GetAcceleration(voltage), std::cos(model.GetPosition()),
          std::sin(model.GetPosition())});

      model.Update(voltage, T);
      voltage -= kUstep * T;
    }
  }

  // Fast forward
  if (movements.test(Movements::kFastForward)) {
    model.Reset();
    voltage = 0_V;
    for (int i = 0; i < (kTestDuration / T).value(); ++i) {
      fastForward.emplace_back(sysid::PreparedData{
          i * T, voltage.value(), model.GetPosition(), model.GetVelocity(), T,
          model.GetAcceleration(voltage), std::cos(model.GetPosition()),
          std::sin(model.GetPosition())});

      model.Update(voltage, T);
      voltage = kUmax;
    }
  }

  // Fast backward
  if (movements.test(Movements::kFastBackward)) {
    model.Reset();
    voltage = 0_V;
    for (int i = 0; i < (kTestDuration / T).value(); ++i) {
      fastBackward.emplace_back(sysid::PreparedData{
          i * T, voltage.value(), model.GetPosition(), model.GetVelocity(), T,
          model.GetAcceleration(voltage), std::cos(model.GetPosition()),
          std::sin(model.GetPosition())});

      model.Update(voltage, T);
      voltage = -kUmax;
    }
  }

  return storage;
}

#if 0
/**
 * Asserts that two arrays are equal.
 *
 * @param expected The expected array.
 * @param actual The actual array.
 * @param tolerances The tolerances for the element comparisons.
 */
void ExpectArrayNear(std::span<const double> expected,
                     std::span<const double> actual,
                     std::span<const double> tolerances) {
  // Check size
  const size_t size = expected.size();
  EXPECT_EQ(size, actual.size());
  EXPECT_EQ(size, tolerances.size());

  // Check elements
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerances[i]) << "where i = " << i;
  }
}
#endif

#if 0
static std::vector<double> BuildFeedforwardGainsArray(const sysid::FeedforwardGains& ff, const sysid::AnalysisType& type) {
  if (type == sysid::analysis::kSimple) {
    return {ff.Ks.gain, ff.Kv.gain, ff.Ka.gain};
  } else if (type == sysid::analysis::kElevator) {
    return {ff.Ks.gain, ff.Kv.gain, ff.Ka.gain, ff.Kg.gain};
  } else if (type == sysid::analysis::kArm) {
    return {ff.Ks.gain, ff.Kv.gain, ff.Ka.gain, ff.Kg.gain, ff.offset.gain};
  } else {
    throw std::runtime_error("NotImplemented!");
  }
}
#endif

/**
 * Computes the decimal percent error between 2 doubles.
 * @param actual The actual value.
 * @param expected The expected value.
 * @return The percent error in decimal form (0.5 is 50%)
*/
static constexpr double PercentErrorDecimal(const double actual, const double expected) {
  return (actual - expected) / expected;
}

/**
 * Computes if a FeedforwardGain is impossible based its value compared to it's possible range of values, e.g. Ks < 0 is erroneous.
 * 
 * @param ffGain The FeedforwardGain to test against.
 * @param gainIndex The corresponding index of the gain in typical order, {Ks, Kv, Ka, [Kg], [offset]} where [] denotes optional (conditional)
*/
static bool IsOutOfRangeFeedforwardGain(const sysid::FeedforwardGain& ffGain, const size_t gainIndex) {
  return (gainIndex == 0 && ffGain.gain < 0)  // Ks
    || (gainIndex == 1 && ffGain.gain < 0 )   // Kv
    || (gainIndex == 2 && ffGain.gain <= 0)   // Ka
    || (gainIndex == 3 && ffGain.gain < 0);   // Kg
}

/**
 * Asserts success if every FeedforwardGain in FeedforwardGains if marked as valid, must fall within a percent error tolerance to the expected gain,
 * and if invalid, must fall outside the tolerance. Gains that contain NaNs are expected to be marked as invalid.
 * 
 * @param actualGains The calculated feedforward gains.
 * @param expectedGains The expected feedforward gains.
 * @param tolerances The percent tolerances (decimal) for the coefficient comparisons.
 * @param absTolerancesOnZero The absolute tolerances to use if the expected gain is zero (percent-value does not work when expected=0).
 */
static testing::AssertionResult AssertFeedforwardGains(const sysid::FeedforwardGains& actualGains,
                                                      const sysid::AnalysisType& analysisType,
                                                      std::span<const double> expectedGains,
                                                      std::span<const double> percentTolerances,
                                                      std::span<const double> absTolerancesOnZero) {
  // Size of expectedGains should be equal to the number of independent variables in the analysis
  assert(analysisType.independentVariables == expectedGains.size());
  constexpr std::array gainNames{"Ks", "Kv", "Ka", "Kg", "offset"};
  const std::array gains{actualGains.Ks, actualGains.Kv, actualGains.Ka, actualGains.Kg, actualGains.offset};

  auto result = testing::AssertionSuccess();
  for (size_t i = 0; i < analysisType.independentVariables; ++i) {
    const sysid::FeedforwardGain& ffGain = gains[i];
    const double expectedGain = expectedGains[i];
    // If expectedGain == 0, then we can't do percent-value on it, so fall-back to absolute error and tolerances
    const bool useAbsDiff = expectedGain == 0;
    const double tolerance = useAbsDiff ? absTolerancesOnZero[i] : percentTolerances[i];
    const double absError = useAbsDiff ? std::abs(ffGain.gain - expectedGain) : std::abs(PercentErrorDecimal(ffGain.gain, expectedGain));

    // Valid gain, error is at or below tolerance
    if (ffGain.isValidGain && absError <= tolerance) {
      continue;
    }
    
    // Invalid gain, error is greater than tolerance or gain is NaN
    if (!ffGain.isValidGain && (absError > tolerance || std::isnan(ffGain.gain) || IsOutOfRangeFeedforwardGain(ffGain, i))) {
      continue;
    }

    // If the result is still a success, make it a failure
    if (result) {
      result = !result;
    }

    result << "\n" << gainNames[i] << ":\n";
    result << "  expected " << expectedGain << ",\n";
    result << "  actual " << ffGain.gain << ",\n";
    result << "  expected-error " << fmt::format("{0} {1:f}{2},\n", ffGain.isValidGain ? "<=" : ">", useAbsDiff ? tolerance : tolerance * 100, useAbsDiff ? "" : "%");
    result << "  actual-error " << fmt::format("{0:f}{1}\n", useAbsDiff ? absError : absError * 100, useAbsDiff ? "" : "%");
  }

  return result;
}

/**
 * @tparam Model The model type.
 * @param model The simulation model.
 * @param type The analysis type.
 * @param expectedGains The expected feedforward gains.
 * @param percentTolerances The percentage tolerances for the coefficient comparisons.
 * @param absTolerancesOnZero The absolute tolerances to use if the expected gain is zero (percent-value does not work when expected=0).
 */
template <typename Model>
void RunTests(Model& model, const sysid::AnalysisType& type,
              std::span<const double> expectedGains,
              std::span<const double> percentTolerances,
              std::span<const double> absTolerancesOnZero) {
  // Iterate through all combinations of movements
  for (int movements = 0; movements < kMovementCombinations; ++movements) {
    auto ff =
        sysid::CalculateFeedforwardGains(CollectData(model, movements), type);

    // fmt::print("Movements: {}\n", movements);
    // fmt::print("Ks: {}, Kv: {}, Ka: {}\n", ff.Ks.gain, ff.Kv.gain,
    //             ff.Ka.gain);
    // fmt::print("KsGood: {}, KvGood: {}, KaGood: {}\n", ff.Ks.isValidGain,
    //             ff.Kv.isValidGain, ff.Ka.isValidGain);
    // fmt::print("KsE: {0}:{1:.3f}%, KvE: {2}:{3:.3f}%, KaE: {4}:{5:.3f}%\n\n",
    //             expectedGains[0], PercentErrorDecimal(ff.Ks.gain, expectedGains[0]) * 100,
    //             expectedGains[1], PercentErrorDecimal(ff.Kv.gain, expectedGains[1]) * 100,
    //             expectedGains[2], PercentErrorDecimal(ff.Ka.gain, expectedGains[2]) * 100);

    EXPECT_TRUE(AssertFeedforwardGains(ff, type, expectedGains, percentTolerances, absTolerancesOnZero));
  }
}

}  // namespace

TEST(FeedforwardAnalysisTest, Arm) {
  {
    constexpr double Ks = 1.01;
    constexpr double Kv = 3.060;
    constexpr double Ka = 0.327;
    constexpr double Kg = 0.211;

    for (const auto& offset : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
      sysid::ArmSim model{Ks, Kv, Ka, Kg, offset};

      RunTests(model, sysid::analysis::kArm, {{Ks, Kv, Ka, Kg, offset}}, {{8e-2, 8e-2, 8e-2, 8e-2, 8e-2}}, {{1e-4, 1e-4, 1e-4, 1e-4, 2e-2}});
    }
  }

  {
    constexpr double Ks = 0.547;
    constexpr double Kv = 0.0693;
    constexpr double Ka = 0.1170;
    constexpr double Kg = 0.122;

    for (const auto& offset : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
      sysid::ArmSim model{Ks, Kv, Ka, Kg, offset};

      RunTests(model, sysid::analysis::kArm, {{Ks, Kv, Ka, Kg, offset}}, {{8e-2, 8e-2, 8e-2, 8e-2, 8e-2}}, {{1e-4, 1e-4, 1e-4, 1e-4, 2e-2}});
    }
  }
}

TEST(FeedforwardAnalysisTest, Elevator) {
  {
    constexpr double Ks = 1.01;
    constexpr double Kv = 3.060;
    constexpr double Ka = 0.327;
    constexpr double Kg = -0.211;

    sysid::ElevatorSim model{Ks, Kv, Ka, Kg};

    RunTests(model, sysid::analysis::kElevator, {{Ks, Kv, Ka, Kg}},
             {{8e-2, 8e-2, 8e-2, 8e-2}}, {{1e-4, 1e-4, 1e-4, 1e-4}});
  }

  {
    constexpr double Ks = 0.547;
    constexpr double Kv = 0.0693;
    constexpr double Ka = 0.1170;
    constexpr double Kg = -0.122;

    sysid::ElevatorSim model{Ks, Kv, Ka, Kg};

    RunTests(model, sysid::analysis::kElevator, {{Ks, Kv, Ka, Kg}},
             {{8e-2, 8e-2, 8e-2, 8e-2}}, {{1e-4, 1e-4, 1e-4, 1e-4}});
  }
}

TEST(FeedforwardAnalysisTest, Simple) {
  {
    constexpr double Ks = 1.01;
    constexpr double Kv = 3.060;
    constexpr double Ka = 0.327;

    sysid::SimpleMotorSim model{Ks, Kv, Ka};

    RunTests(model, sysid::analysis::kSimple, {{Ks, Kv, Ka}},
             {{8e-2, 8e-2, 8e-2}}, {{1e-4, 1e-4, 1e-4}});
  }

  {
    constexpr double Ks = 0.547;
    constexpr double Kv = 0.0693;
    constexpr double Ka = 0.1170;

    sysid::SimpleMotorSim model{Ks, Kv, Ka};

    RunTests(model, sysid::analysis::kSimple, {{Ks, Kv, Ka}},
             {{8e-2, 8e-2, 8e-2}}, {{1e-4, 1e-4, 1e-4}});
  }
}
