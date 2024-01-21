// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "sysid/analysis/FeedforwardAnalysis.h"

#include <array>
#include <bitset>
#include <cmath>

#include <Eigen/Eigenvalues>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <units/math.h>
#include <units/time.h>

#include "sysid/analysis/AnalysisManager.h"
#include "sysid/analysis/FilteringUtils.h"
#include "sysid/analysis/OLS.h"

namespace sysid {

/**
 * Populates OLS data for the following models:
 *
 * Simple, Drivetrain, DrivetrainAngular:
 *
 *   (xₖ₊₁ − xₖ)/τ = αxₖ + βuₖ + γ sgn(xₖ)
 *
 * Elevator:
 *
 *   (xₖ₊₁ − xₖ)/τ = αxₖ + βuₖ + γ sgn(xₖ) + δ
 *
 * Arm:
 *
 *   (xₖ₊₁ − xₖ)/τ = αxₖ + βuₖ + γ sgn(xₖ) + δ cos(angle) + ε sin(angle)
 *
 * OLS performs best with the noisiest variable as the dependent variable, so we
 * regress acceleration in terms of the other variables.
 *
 * @param d List of characterization data.
 * @param type Type of system being identified.
 * @param X Vector representation of X in y = Xβ.
 * @param y Vector representation of y in y = Xβ.
 */
static void PopulateOLSData(const std::vector<PreparedData>& d,
                            const AnalysisType& type,
                            Eigen::Block<Eigen::MatrixXd> X,
                            Eigen::VectorBlock<Eigen::VectorXd> y) {
  // Fill in X and y row-wise
  for (size_t sample = 0; sample < d.size(); ++sample) {
    const auto& pt = d[sample];

    // Set the velocity term (for α)
    X(sample, 0) = pt.velocity;

    // Set the voltage term (for β)
    X(sample, 1) = pt.voltage;

    // Set the intercept term (for γ)
    X(sample, 2) = std::copysign(1, pt.velocity);

    // Set test-specific variables
    if (type == analysis::kElevator) {
      // Set the gravity term (for δ)
      X(sample, 3) = 1.0;
    } else if (type == analysis::kArm) {
      // Set the cosine and sine terms (for δ and ε)
      X(sample, 3) = pt.cos;
      X(sample, 4) = pt.sin;
    }

    // Set the dependent variable (acceleration)
    y(sample) = pt.acceleration;
  }
}

/**
 * Finds the worst fit coefficients for OLS and the impacted gains.
 *
 * @param X The sliced data formed from all collected data in matrix form for OLS.
 * @param sliceIndexes The currently sliced indexes for X
 * @param type The analysis type.
 */
static OLSDataQuality CheckOLSDataQuality(const Eigen::MatrixXd& X, const std::vector<int>& sliceIndexes, const AnalysisType& type) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver{X.transpose() * X};
  const Eigen::VectorXd& eigvals = eigSolver.eigenvalues();
  const Eigen::MatrixXd& eigvecs = eigSolver.eigenvectors();

  constexpr double threshold = 10.0;
  OLSDataQuality dataQuality;

  // For n x n matrix XᵀX, need n nonzero eigenvalues for good fit
  for (int row = 0; row < eigvals.rows(); ++row) {
    // Find row of eigenvector with largest magnitude. This determines the
    // primary regression variable that corresponds to the eigenvalue.
    int maxIndex;
    double maxCoeff = eigvecs.col(row).cwiseAbs().maxCoeff(&maxIndex);


    int remappedMaxIndex = sliceIndexes[maxIndex];
    // Check whether the eigenvector component along the regression variable's
    // direction is below the threshold. If it is, the regression variable's fit
    // is bad.
    double absEigenVecComponent = std::abs(eigvals(row) * maxCoeff);
    if (absEigenVecComponent <= threshold) {
      // Below threshold and is worse than previous component
      if (!dataQuality.hasWorstCoeff || absEigenVecComponent < dataQuality.minAbsEigVecComponent) {
        dataQuality.minAbsEigVecComponent = absEigenVecComponent;
        dataQuality.hasWorstCoeff = true;
        dataQuality.worstFitCoeffIndex = remappedMaxIndex;
      }

      auto& badGains = dataQuality.badGains;
      // Fit for α is bad
      if (remappedMaxIndex == 0) {
        // Affects Kv
        badGains.set(1);
      }

      // Fit for β is bad
      if (remappedMaxIndex == 1) {
        // Affects all gains
        badGains.set();
      }

      // Fit for γ is bad
      if (remappedMaxIndex == 2) {
        // Affects Ks
        badGains.set(0);
      }

      // Fit for δ is bad
      if (remappedMaxIndex == 3) {
        if (type == analysis::kElevator) {
          // Affects Kg
          badGains.set(3);
        } else if (type == analysis::kArm) {
          // Affects Kg and offset
          badGains.set(3);
          badGains.set(4);
        }
      }

      // Fit for ε is bad
      if (remappedMaxIndex == 4) {
        // Affects Kg and offset
        badGains.set(3);
        badGains.set(4);
      }
    }
  }

  return dataQuality;
}

static std::optional<ptrdiff_t> GetIndexOf(const std::vector<int>& vec, const int& val) {
  const ptrdiff_t index = std::distance(vec.begin(), std::find(vec.begin(), vec.end(), val));
  if (index >= (ptrdiff_t)vec.size()) {
    return {};
  }

  return index;
}

FeedforwardGains CalculateFeedforwardGains(const Storage& data, const AnalysisType& type, bool throwOnBadData) {
  // Iterate through the data and add it to our raw vector.
  const auto& [slowForward, slowBackward, fastForward, fastBackward] = data;

  const auto size = slowForward.size() + slowBackward.size() +
                    fastForward.size() + fastBackward.size();

  // Create a raw vector of doubles with our data in it.
  Eigen::MatrixXd X{size, type.independentVariables};
  Eigen::VectorXd y{size};

  int rowOffset = 0;
  PopulateOLSData(slowForward, type,
                  X.block(rowOffset, 0, slowForward.size(), X.cols()),
                  y.segment(rowOffset, slowForward.size()));

  rowOffset += slowForward.size();
  PopulateOLSData(slowBackward, type,
                  X.block(rowOffset, 0, slowBackward.size(), X.cols()),
                  y.segment(rowOffset, slowBackward.size()));

  rowOffset += slowBackward.size();
  PopulateOLSData(fastForward, type,
                  X.block(rowOffset, 0, fastForward.size(), X.cols()),
                  y.segment(rowOffset, fastForward.size()));

  rowOffset += fastForward.size();
  PopulateOLSData(fastBackward, type,
                  X.block(rowOffset, 0, fastBackward.size(), X.cols()),
                  y.segment(rowOffset, fastBackward.size()));

  // Check quality of collected data
  // if (throwOnBadData) {
  //   CheckOLSDataQuality(X, type);
  // }

  std::vector<int> remainingIndexes(type.independentVariables);
  // Fill remainingIndexes with all possible indexes
  std::iota(std::begin(remainingIndexes), std::end(remainingIndexes), 0);

  OLSResult ols;
  FeedforwardGains ffGains;

  while (!remainingIndexes.empty()) {
    const auto slicedX = X(Eigen::placeholders::all, remainingIndexes);
    const auto olsQuality = CheckOLSDataQuality(slicedX, remainingIndexes, type);

    ols = OLS(slicedX, y);

    if (!olsQuality.hasWorstCoeff) {
      // Didn't find worst coeff, which means we can exit early
      break;
    }
    
    const auto& badGains = olsQuality.badGains;
    // Ks, Kg, and offset are ignored since it is possible that they have no contribution, which is indistinguishable from poor sampling.
    // Kv, Ka are not ignored, which are at bitset indexes 1 and 2.

    // Kv
    if (badGains.test(1)) {
      // α = -Kv/Ka
      // β = 1/Ka

      // Remap index to sliced ols fit by finding index of original in remainingIndexes
      const auto αPos = GetIndexOf(remainingIndexes, 0);
      const auto βPos = GetIndexOf(remainingIndexes, 1);
      if (αPos.has_value() && βPos.has_value()) {
        double α = ols.coeffs[αPos.value()];
        double β = ols.coeffs[βPos.value()];

        ffGains.Kv.gain = -α / β;
      }

      ffGains.Kv.isValidGain = false;
      ffGains.Kv.errorMessage = fmt::format(
        "Insufficient samples to compute Kv. Ensure the data has at least 2 steady-state velocity events to separate Ks from Kv."
      );
    }

    // Ka
    if (badGains.test(2)) {
      // β = 1/Ka
      const auto βPos = GetIndexOf(remainingIndexes, 1);
      if (βPos.has_value()) {
        double β = ols.coeffs[βPos.value()];

        // Ka = 1/β
        ffGains.Ka.gain = 1 / β;
      }

      ffGains.Ka.isValidGain = false;
      ffGains.Ka.errorMessage = fmt::format(
        "Insufficient samples to compute Ka. Ensure the data has at least 1 acceleration event to find Ka."
      );
    }

    // Remove worst fit coefficient index from remaining indexes
    std::erase(remainingIndexes, olsQuality.worstFitCoeffIndex);
  }

  // Fill OLSResult into FeedforwardGains
  ffGains.olsResult = ols;

  // Calculate remaining (good) feedforward gains
  //
  // See docs/ols-derivations.md for more details.
  {
    // dx/dt = -Kv/Ka x + 1/Ka u - Ks/Ka sgn(x)
    // dx/dt = αx + βu + γ sgn(x)

    // α = -Kv/Ka
    // β = 1/Ka
    // γ = -Ks/Ka
    const auto αPos = GetIndexOf(remainingIndexes, 0);
    const auto βPos = GetIndexOf(remainingIndexes, 1);
    const auto γPos = GetIndexOf(remainingIndexes, 2);

    // Ks = -γ/β
    if (βPos.has_value() && γPos.has_value()) {
      double β = ols.coeffs[βPos.value()];
      double γ = ols.coeffs[γPos.value()];

      ffGains.Ks.gain = -γ / β;
    }

    // Kv = -α/β
    if (αPos.has_value() && βPos.has_value()) {
      double α = ols.coeffs[αPos.value()];
      double β = ols.coeffs[βPos.value()];

      ffGains.Kv.gain = -α / β;
    }

    // Ka = 1/β
    if (βPos.has_value()) {
      double β = ols.coeffs[βPos.value()];
      ffGains.Ka.gain = 1 / β;
    }

    if (type == analysis::kElevator) {
      // dx/dt = -Kv/Ka x + 1/Ka u - Ks/Ka sgn(x) - Kg/Ka
      // dx/dt = αx + βu + γ sgn(x) + δ

      // δ = -Kg/Ka
      const auto δPos = GetIndexOf(remainingIndexes, 3);
      if (δPos.has_value() && βPos.has_value()) {
        double δ = ols.coeffs[δPos.value()];
        double β = ols.coeffs[βPos.value()];

        // Kg = -δ/β
        ffGains.Kg.gain = -δ / β;
      }
    }

    if (type == analysis::kArm) {
      // dx/dt = -Kv/Ka x + 1/Ka u - Ks/Ka sgn(x)
      //           - Kg/Ka cos(offset) cos(angle)                   NOLINT
      //           + Kg/Ka sin(offset) sin(angle)                   NOLINT
      // dx/dt = αx + βu + γ sgn(x) + δ cos(angle) + ε sin(angle)   NOLINT

      // δ = -Kg/Ka cos(offset)
      // ε = Kg/Ka sin(offset)
      const auto δPos = GetIndexOf(remainingIndexes, 3);
      const auto εPos = GetIndexOf(remainingIndexes, 4);

      if (δPos.has_value() && εPos.has_value() && βPos.has_value()) {
        double δ = ols.coeffs[δPos.value()];
        double ε = ols.coeffs[εPos.value()];
        double β = ols.coeffs[βPos.value()];

        // Kg = hypot(δ, ε)/β      NOLINT
        // offset = atan2(ε, -δ)   NOLINT
        ffGains.Kg.gain = std::hypot(δ, ε) / β;
        ffGains.offset.gain = std::atan2(ε, -δ);
      }
    }
  }

  // Ks, Kv, and Kg senisible gains checking, and fills descriptors
  {
    auto& Ks = ffGains.Ks;
    Ks.descriptor = "Voltage needed to overcome static friction.";
    if (Ks.isValidGain && Ks.gain < 0) {
      Ks.isValidGain = false;
      Ks.errorMessage = fmt::format("Calculated Ks gain of: {0:.3f} is erroneous! Ks should be >= 0.", Ks.gain);
    }

    auto& Kv = ffGains.Kv;
    Kv.descriptor =
          "Voltage needed to hold/cruise at a constant velocity while "
          "overcoming the counter-electromotive force and any additional "
          "friction.";
    if (Kv.isValidGain && Kv.gain < 0) {
      Kv.isValidGain = false;
      Kv.errorMessage = fmt::format(
          "Calculated Kv gain of: {0:.3f} is erroneous! Kv should be >= 0.", Kv.gain);
    }

    auto& Ka = ffGains.Ka;
    Ka.descriptor = "Voltage needed to induce a given acceleration in the motor shaft.";
    if (Ka.isValidGain && Ka.gain <= 0) {
      Ka.isValidGain = false;
      Ka.errorMessage = fmt::format(
          "Calculated Ka gain of: {0:.3f} is erroneous! Ka should be > 0.", Ka.gain);
    }
  }

  if (type == analysis::kElevator || type == analysis::kArm) {
    auto& Kg = ffGains.Kg;
    Kg.descriptor = "Voltage needed to counteract the force of gravity.";
    if (Kg.isValidGain && Kg.gain < 0) {
      Kg.isValidGain = false;
      Kg.errorMessage = fmt::format("Calculated Kg gain of: {0:.3f} is erroneous! Kg should be >= 0.", Kg.gain);
    }

    // Elevator analysis only requires Kg
    if (type == analysis::kElevator) {
      return ffGains;
    } else {
      // Arm analysis requires Kg and an angle offset
      auto& offset = ffGains.offset;
      offset.descriptor =
              "This is the angle offset which, when added to the angle "
              "measurement, zeroes it out when the arm is horizontal. This is "
              "needed for the arm feedforward to work.";
      return ffGains;
    }
  }

  return ffGains;
}

}  // namespace sysid
