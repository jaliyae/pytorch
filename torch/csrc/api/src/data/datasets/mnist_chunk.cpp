/*
#include <torch/data/datasets/mnist_chunk.h>

#include <torch/data/example.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
namespace {
constexpr uint32_t kTrainSize = 60000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kImageMagicNumber = 2051;
constexpr uint32_t kTargetMagicNumber = 2049;
constexpr uint32_t kImageRows = 28;
constexpr uint32_t kImageColumns = 28;
constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
constexpr const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
constexpr const char* kTestTargetsFilename = "t10k-labels-idx1-ubyte";

bool check_is_little_endian() {
  const uint32_t word = 1;
  return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness(uint32_t value) {
  return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
      ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32(std::ifstream& stream) {
  static const bool is_little_endian = check_is_little_endian();
  uint32_t value;
  AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
  return is_little_endian ? flip_endianness(value) : value;
}

uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
  const auto value = read_int32(stream);
  // clang-format off
  AT_CHECK(value == expected,
      "Expected to read number ", expected, " but found ", value, " instead");
  // clang-format on
  return value;
}

std::string join_paths(std::string head, std::string tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += std::move(tail);
  return head;
}

Tensor read_images(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  AT_CHECK(images, "Error opening images file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  // From http://yann.lecun.com/exdb/mnist/
  expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  auto tensor =
      torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  return tensor.to(torch::kFloat32).div_(255);
}

Tensor read_targets(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  std::ifstream targets(path, std::ios::binary);
  AT_CHECK(targets, "Error opening targets file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  expect_int32(targets, kTargetMagicNumber);
  expect_int32(targets, count);

  auto tensor = torch::empty(count, torch::kByte);
  targets.read(reinterpret_cast<char*>(tensor.data_ptr()), count);
  return tensor.to(torch::kInt64);
}
} // namespace

template <typename ChunkSampler, typename ExampleSampler>
MNIST_CHUNK<ChunkSampler, ExampleSampler>::MNIST_CHUNK(
    const std::string& root,
    ChunkSampler chunk_sampler,
    ExampleSampler example_sampler,
    size_t num_prefetch_threads,
    Mode mode)
    : datasets::ChunkDataSet<
          MNIST_CHUNK<ChunkSampler, ExampleSampler>,
          std::vector<Example<>>,
          ChunkSampler,
          ExampleSampler>(num_prefetch_threads),
      chunk_sampler_(std::move(chunk_sampler)),
      example_sampler_(std::move(example_sampler)),
      images_(read_images(root, mode == Mode::kTrain)),
      targets_(read_targets(root, mode == Mode::kTrain)) {}

template <typename ChunkSampler, typename ExampleSampler>
std::vector<Example<>> MNIST_CHUNK<ChunkSampler, ExampleSampler>::read_chunk(
    size_t chunk_index) {
  return {{images_[chunk_index], targets_[chunk_index]}};
}

template <typename ChunkSampler, typename ExampleSampler>
bool MNIST_CHUNK<ChunkSampler, ExampleSampler>::is_train() const noexcept {
  return images_.size(0) == kTrainSize;
}

template <typename ChunkSampler, typename ExampleSampler>
const Tensor& MNIST_CHUNK<ChunkSampler, ExampleSampler>::images() const {
  return images_;
}

template <typename ChunkSampler, typename ExampleSampler>
const Tensor& MNIST_CHUNK<ChunkSampler, ExampleSampler>::targets() const {
  return targets_;
}

template <typename ChunkSampler, typename ExampleSampler>
 size_t MNIST_CHUNK<ChunkSampler, ExampleSampler>::get_chunk_count(){
  return is_train() ? kTrainSize : kTestSize;
}

} // namespace datasets
} // namespace data
} // namespace torch
*/