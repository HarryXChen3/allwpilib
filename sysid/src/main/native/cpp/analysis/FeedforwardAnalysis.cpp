// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "sysid/analysis/FeedforwardAnalysis.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <string>
#include <vector>
#include <numeric>

#include <Eigen/Eigenvalues>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <units/math.h>
#include <units/time.h>

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
 * Returns index of worst fit coefficient for OLS, if any were bad.
 *
 * @param X The data formed from all collected data in matrix form for OLS.
 * @return Index of worst fit coefficient for OLS, if any were bad.
 */
static std::optional<int> FindWorstBadOLSCoeff(const Eigen::MatrixXd& X) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver{X.transpose() * X};

  int minIndex;
  double minCoeff =
      (eigSolver.eigenvectors() * eigSolver.eigenvalues().asDiagonal())
          // Find row of each eigenvector with largest magnitude. This
          // determines the primary regression variable that corresponds to each
          // eigenvalue.
          .cwiseAbs()
          .colwise()
          .maxCoeff()
          // Find the column with the smallest eigenvector component
          .minCoeff(&minIndex);

  // If the eigenvector component along the regression variable's direction is
  // below the threshold, the regression variable's fit is bad
  if (minCoeff < 10.0) {
    return minIndex;
  } else {
    return {};
  }
}

FeedforwardGains CalculateFeedforwardGains(const Storage& data,
                                           const AnalysisType& type) {
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

  // Fill remainingIndices with all possible indices
  std::vector<int> remainingIndices(type.independentVariables);
  std::iota(std::begin(remainingIndices), std::end(remainingIndices), 0);

  OLSResult ols;
  FeedforwardGains ffGains;

  Eigen::VectorXd allOLSCoeffs{type.independentVariables};

  while (!remainingIndices.empty()) {
    const auto slicedX = X(Eigen::placeholders::all, remainingIndices);
    auto badSubindex = FindWorstBadOLSCoeff(slicedX);

    ols = OLS(slicedX, y);

    if (!badSubindex.has_value()) {
      // Didn't find worst coeff, which means we can exit early
      for (size_t i = 0; i < remainingIndices.size(); ++i) {
        allOLSCoeffs[remainingIndices[i]] = ols.coeffs[i];
      }
      break;
    }

    int badIndex = remainingIndices[badSubindex.value()];

    // Ks, Kg, and offset are ignored since it is possible that they have no
    // contribution, which is indistinguishable from poor sampling. Kv, Ka are
    // not ignored, which are at bitset indices 1 and 2.

    allOLSCoeffs[badIndex] = ols.coeffs[badSubindex.value()];

    if (badIndex == 0) {
      ffGains.Kv.isValidGain = false;
      ffGains.Kv.errorMessage =
          "Insufficient samples to compute Kv. Ensure the data has at least 2 "
          "steady-state velocity events to separate Ks from Kv.";
    } else if (badIndex == 1) {
      FeedforwardGain badGain{
          .isValidGain = false,
          .errorMessage =
              "Insufficient samples to compute any gains. Ensure the data has:\n\n"
              "  * at least 2 steady-state velocity events to separate Ks from Kv\n"
              "  * at least 1 acceleration event to find Ka\n"
              "  * for elevators, enough vertical motion to measure gravity\n"
              "  * for arms, enough range of motion to measure gravity and encoder offset\n"};
      ffGains.Ks = badGain;
      ffGains.Kv = badGain;
      ffGains.Ka = badGain;
      ffGains.Kg = badGain;
      ffGains.offset = badGain;
    }

    // Remove worst fit coefficient index from remaining indices
    std::erase(remainingIndices, badIndex);
  }

  // Fill OLSResult into FeedforwardGains
  ffGains.olsResult = ols;

  // Calculate feedforward gains
  //
  // See docs/ols-derivations.md for more details.
  {
    // dx/dt = -Kv/Ka x + 1/Ka u - Ks/Ka sgn(x)
    // dx/dt = αx + βu + γ sgn(x)

    // α = -Kv/Ka
    // β = 1/Ka
    // γ = -Ks/Ka
    double α = allOLSCoeffs[0];
    double β = allOLSCoeffs[1];
    double γ = allOLSCoeffs[2];

    // Ks = -γ/β
    ffGains.Ks.gain = -γ / β;

    // Kv = -α/β
    ffGains.Kv.gain = -α / β;

    // Ka = 1/β
    ffGains.Ka.gain = 1 / β;

    if (type == analysis::kElevator) {
      // dx/dt = -Kv/Ka x + 1/Ka u - Ks/Ka sgn(x) - Kg/Ka
      // dx/dt = αx + βu + γ sgn(x) + δ

      // δ = -Kg/Ka
      double δ = allOLSCoeffs[3];

      // Kg = -δ/β
      ffGains.Kg.gain = -δ / β;
    }

    if (type == analysis::kArm) {
      // dx/dt = -Kv/Ka x + 1/Ka u - Ks/Ka sgn(x)
      //           - Kg/Ka cos(offset) cos(angle)                   NOLINT
      //           + Kg/Ka sin(offset) sin(angle)                   NOLINT
      // dx/dt = αx + βu + γ sgn(x) + δ cos(angle) + ε sin(angle)   NOLINT

      // δ = -Kg/Ka cos(offset)
      // ε = Kg/Ka sin(offset)
      double δ = allOLSCoeffs[3];
      double ε = allOLSCoeffs[4];

      // Kg = hypot(δ, ε)/β      NOLINT
      // offset = atan2(ε, -δ)   NOLINT
      ffGains.Kg.gain = std::hypot(δ, ε) / β;
      ffGains.offset.gain = std::atan2(ε, -δ);
    }
  }

  // Ks, Kv, and Kg senisible gains checking, and fills descriptors
  {
    auto& Ks = ffGains.Ks;
    Ks.descriptor = "Voltage needed to overcome static friction.";
    if (Ks.isValidGain && Ks.gain < 0) {
      Ks.isValidGain = false;
      Ks.errorMessage = fmt::format(
          "Calculated Ks gain of: {0:.3f} is erroneous! Ks should be >= 0.",
          Ks.gain);
    }

    auto& Kv = ffGains.Kv;
    Kv.descriptor =
        "Voltage needed to hold/cruise at a constant velocity while "
        "overcoming the counter-electromotive force and any additional "
        "friction.";
    if (Kv.isValidGain && Kv.gain < 0) {
      Kv.isValidGain = false;
      Kv.errorMessage = fmt::format(
          "Calculated Kv gain of: {0:.3f} is erroneous! Kv should be >= 0.",
          Kv.gain);
    }

    auto& Ka = ffGains.Ka;
    Ka.descriptor =
        "Voltage needed to induce a given acceleration in the motor shaft.";
    if (Ka.isValidGain && Ka.gain <= 0) {
      Ka.isValidGain = false;
      Ka.errorMessage = fmt::format(
          "Calculated Ka gain of: {0:.3f} is erroneous! Ka should be > 0.",
          Ka.gain);
    }
  }

  if (type == analysis::kElevator || type == analysis::kArm) {
    auto& Kg = ffGains.Kg;
    Kg.descriptor = "Voltage needed to counteract the force of gravity.";
    if (Kg.isValidGain && Kg.gain < 0) {
      Kg.isValidGain = false;
      Kg.errorMessage = fmt::format(
          "Calculated Kg gain of: {0:.3f} is erroneous! Kg should be >= 0.",
          Kg.gain);
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
