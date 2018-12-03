#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/datasets/chunk.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <string>

namespace torch {
namespace data {
namespace datasets {
/// The MNIST dataset.
template <typename ChunkSampler = samplers::RandomSampler, typename ExampleSampler= samplers::RandomSampler>
class TORCH_API MNIST_CHUNK : public datasets::ChunkDataSet<MNIST_CHUNK<ChunkSampler, ExampleSampler>, std::vector<Example<>>, ChunkSampler, ExampleSampler> {
 public:
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the MNIST dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
  explicit MNIST_CHUNK(const std::string& root, ChunkSampler chunk_sampler,
      ExampleSampler example_sampler, Mode mode = Mode::kTrain);

  /// Returns true if this is the training subset of MNIST.
  bool is_train() const noexcept;

  /// Returns all images stacked into a single tensor.
  const Tensor& images() const;

  /// Returns all targets stacked into a single tensor.
  const Tensor& targets() const;

  /// Read an entire chunk. A derived class needs to override this method.
  /// This is the only API, other than the constructor that
  std::vector<Example<>> read_chunk(size_t chunk_index);

  /// Returns the chunk sampler for this dataset.
  ChunkSampler get_chunk_sampler(){ return chunk_sampler_;}

  /// Returns the example sampler for this dataset.
  ExampleSampler get_example_sampler() { return example_sampler_;}

  /// returns the number of chunks available in this dataset.
  size_t get_chunk_count();

 private:
  Tensor images_, targets_;
  ChunkSampler chunk_sampler_;
  ExampleSampler example_sampler_;
};
} // namespace datasets
} // namespace data
} // namespace torch
