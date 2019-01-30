#pragma once

#include <torch/data/datasets/stateful.h>
#include <random>

namespace torch {
namespace data {
namespace datasets {

/// Interface for chunk reader, which performs data chunking and reading of
/// entire chunks.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text-file split based on seek
/// positions.
template <typename Chunk = std::vector<Example<>>>
class ChunkDataReader {
 public:
  using ChunkType = Chunk;

  /// Read an entire chunk.
  virtual ChunkType read_chunk(size_t chunk_index) = 0;

  /// Returns the number of chunks available in this reader.
  virtual size_t chunk_count() = 0;

  /// This will clear any internal state associate with this reader.
  virtual void reset() = 0;
};

namespace detail {
/// BatchDataBuffer manages a queue of UnwrappedBatchData. After a new chunk is
/// loaded, BatchDataBuffer splits it into small batches and push them into the
/// queue. When get_batch is called from data loader, it pops cached batches and
/// return. If the cache is empty, it either waits to load more chunks or return
/// null if all chunks are loaded.
template <
    typename UnwrappedBatch = std::vector<Example<>>,
    typename ExampleSampler = samplers::RandomSampler>
class BatchDataBuffer {
 public:
  using UnwrappedBatchType = UnwrappedBatch;
  using BatchType = torch::optional<UnwrappedBatchType>;
  using BatchRequestType = typename ExampleSampler::BatchRequestType;

  BatchDataBuffer(
      size_t num_chunks,
      size_t batch_size,
      ExampleSampler& example_sampler,
      size_t queue_capacity)
      : remaining_chunk_count_(num_chunks),
        batch_size_(batch_size),
        example_sampler_(example_sampler),
        queue_capacity_(queue_capacity),
        stop_(false) {}

  /// Return batch data from the queue. Called from the ChunkDataset main
  /// thread.
  BatchType get_batch() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_read_.wait(lock, [this] {
      // wait till there is available data in the queue or if all chunks are
      // loaded (i.e. the dataset is exhausted for this epoch)
      return (
          this->total_example_count_in_queue_ >= batch_size_ ||
          this->remaining_chunk_count_ == 0);
    });
    if (batch_queue_.empty()) {
      AT_ASSERT(remaining_chunk_count_ == 0);

      // All batches have been retrieved. Return an empty batch.
      return nullopt;
    }

    UnwrappedBatchData batch = std::move(batch_queue_.front());
    batch_queue_.pop();
    if (batch.exception) {
      throw WorkerException(batch.exception);
    }

    total_example_count_in_queue_ -= batch.batch_data.size();
    lock.unlock();
    cv_write_.notify_all();

    return batch.batch_data;
  }

  // skip one chunk
  void skip_chunk() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    AT_ASSERT(remaining_chunk_count_ > 0);
    remaining_chunk_count_--;
    lock.unlock();
    cv_read_.notify_all();
  }

  /// Push preloaded chunks to batch queue. Called from the ChunkDataset worker
  /// threads.
  void add_chunk_data(UnwrappedBatchType data) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_write_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < this->queue_capacity_ || stop_.load();
    });

    if (stop_.load()){
      // When stop_ is true, it means this current thread needs to be tore down.
      // Return without any further processing.
      return;
    }

    auto data_size = data.size();
    auto remaining_size = data_size;
    example_sampler_.reset(data_size);

    auto fill_batch = [&](size_t example_count, UnwrappedBatchType& batch) {
      auto batch_example_indices = this->example_sampler_.next(example_count);
      AT_ASSERT(
          batch_example_indices &&
          batch_example_indices.value().size() == example_count)
      BatchRequestType& indices = batch_example_indices.value();
      for (size_t i : indices) {
        AT_CHECK(i < data_size, "Index out of range");
        batch.emplace_back(std::move(data[i]));
      }
      remaining_size -= example_count;
    };

    if (!batch_queue_.empty()) {
      // if the queue has existing data, and the last batch doesn't have enough
      // examples to fill a batch_size batch, add more example to this batch first.
      auto& batch = batch_queue_.back();
      size_t current_count = batch.batch_data.size();
      if (current_count < batch_size_) {
        auto example_count =
            std::min(remaining_size, batch_size_ - current_count);
        fill_batch(example_count, batch.batch_data);
      }
    }

    // If we still have data remaining after filling the last pushed batch, add
    // them to the queue too.
    while (remaining_size > 0) {
      UnwrappedBatchType current_batch;

      // Allocate the batch memory ahead of time.
      current_batch.reserve(batch_size_);

      auto example_count = std::min(remaining_size, batch_size_);
      fill_batch(example_count, current_batch);
      batch_queue_.emplace(std::move(current_batch));
    }
    total_example_count_in_queue_ += data_size;

    AT_ASSERT(remaining_chunk_count_ > 0);
    remaining_chunk_count_--;

    lock.unlock();
    cv_read_.notify_all();
  }

  /// Push exceptions thrown during preloading into batch queue. Called from
  /// the ChunkDataset worker threads.
  void add_chunk_data(std::exception_ptr e_ptr) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_write_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < this->queue_capacity_ || stop_.load();
    });

    if (stop_.load()){
      // When stop_ is true, it means this current thread needs to be tore down,
      // the batch buffer will be discarded, so no need to enqueue any new
      // exceptions.
      return;
    }

    batch_queue_.emplace(e_ptr);

    AT_ASSERT(remaining_chunk_count_ > 0);
    remaining_chunk_count_--;
    lock.unlock();
    cv_read_.notify_all();
  }

  void stop(){
    stop_ = true;

    // notify all writers, wake them from wait to exit current method.
    cv_write_.notify_all();
  }

  /// count of remaining chunk to be loaded. It is initialized with the total
  /// chunk count and it decreases when a chunk data is retrieved. When this reaches
  /// to 0, no more chunk needs to be loaded.
  size_t remaining_chunk_count_ = 0;

  /// The batch size is needed to create batches from the chunk data. Similar to
  /// regular dataloader where the batches are created with prefetches,
  /// BatchDataBuffer perform the batch creation using the provided batch size.
  size_t batch_size_ = 0;

  /// count of total example stored in the queue
  size_t total_example_count_in_queue_ = 0;

  /// struct that contains a raw unwrapped batch unit. An unwrapped batch unit is
  /// the raw data without 'optional' wrapper. It can be a collection of images,
  /// utterances, e.t.c.
  struct UnwrappedBatchData {
    explicit UnwrappedBatchData(UnwrappedBatchType data) : batch_data(std::move(data)) {}

    explicit UnwrappedBatchData(std::exception_ptr e) : exception(e) {}

    /// batch data to return
    UnwrappedBatchType batch_data;

    /// exception pointer which captures any abnormal exceptions while creating the
    /// batch.
    std::exception_ptr exception;
  };

  /// local cache to store example batches from loaded chunk
  std::queue<UnwrappedBatchData> batch_queue_;

  // sync batch_queue_ update.
  std::mutex queue_mutex_;

  std::condition_variable cv_read_;
  std::condition_variable cv_write_;

  ExampleSampler& example_sampler_;

  // configurable maximun number of elements the queue can hold at one time.
  size_t queue_capacity_;

  // When set to true, it wakes the writer threads from the wait and exit current
  // function call. This is needed when ChunkDataSet.Reset is called while the
  // previous epoch is not exhausted yet. When ChunkDataset is waiting its
  // preloader to finish previous work before tearing down the thread, the
  // preloader could be still waiting for the conditional variable, thus cause
  // the program to hang. This boolean is used to break this waiting condition.
  std::atomic<bool> stop_;
};

/// Select chunks for loading and define a sampling behavior.
/// In a distributed setting, it selects a subset of the chunks depending on the
/// provided num_replicas and rank parameters.
/// The `next()` method of this class needs to be thread-safe as it will be
/// called from different threads during chunk loading.
class ChunkSelector {
 public:
  virtual ~ChunkSelector() = default;
  ChunkSelector(size_t chunk_count, size_t num_replicas = 1, size_t rank = 0)
      : chunk_count_(chunk_count), num_replicas_(num_replicas), rank_(rank) {
    local_chunk_count_ = (size_t)std::ceil(chunk_count_ * 1.0 / num_replicas_);
  }

  /// Get the next chunk index for loading.
  /// Note: this method needs to be thread-safe.
  virtual optional<size_t> next() = 0;

  /// Reset the chunk selector for a new enumeration.
  virtual void reset() = 0;

  /// Set the epoch for the current enumeration. This can be used to alter the
  /// chunk selection and shuffling behavior.
  void set_epoch(size_t epoch) {
    epoch_ = epoch;
  }

  /// Return the number of chunks to be loaded. In the case of distributed
  /// training, this is different to chunk_count as each loader needs to load
  /// only a subset of chunks.
  size_t local_chunk_count() {
    return local_chunk_count_;
  }

 protected:
  size_t epoch_{0};
  size_t chunk_count_;
  size_t num_replicas_;
  size_t rank_;
  size_t local_chunk_count_;
};

/// Select chunks randomly. The chunk order shuffled at each `reset()` call.
class RandomChunkSelector : public ChunkSelector {
 public:
  RandomChunkSelector(
      size_t chunk_count,
      size_t num_replicas = 1,
      size_t rank = 0)
      : ChunkSelector(chunk_count, num_replicas, rank) {
    size_t index_count =
        num_replicas_ == 1 ? chunk_count_ : local_chunk_count_ * num_replicas_;
    all_indices_.resize(index_count);
    if (num_replicas_ > 1)
      for (size_t i = 0; i < index_count; ++i) {
        all_indices_[i] =
            i % chunk_count_; // we are adding some more chunks to make all
                              // replicas to have the same number of chunks.
      }
    else {
      std::iota(std::begin(all_indices_), std::end(all_indices_), 0);
    }
  }

  optional<size_t> next() { 
    AT_CHECK(
        !chunk_indices_.empty(),
        "reset() needs to be called before calling next().");
    size_t idx = current_index_.fetch_add(1, std::memory_order_relaxed);
    if (idx < chunk_indices_.size()) {
      return chunk_indices_[idx];
    } else {
      return nullopt;
    }
  }

  void reset() override {
    std::minstd_rand rand(epoch_);
    std::shuffle(all_indices_.begin(), all_indices_.end(), rand);
    chunk_indices_.clear();
    chunk_indices_.reserve(local_chunk_count_);
    auto begin_index_iter = all_indices_.begin() + rank_ * local_chunk_count_;
    auto end_index_iter = begin_index_iter + local_chunk_count_;
    std::copy(begin_index_iter, end_index_iter, back_inserter(chunk_indices_));
    current_index_ = 0;
  }

 private:
  std::vector<size_t> all_indices_;
  std::vector<size_t> chunk_indices_;
  std::atomic<size_t> current_index_{0};
};

/// Select chunks sequentially. 
class SequentialChunkSelector : public ChunkSelector {
 public:
  SequentialChunkSelector(
      size_t chunk_count,
      size_t num_replicas = 1,
      size_t rank = 0)
      : ChunkSelector(chunk_count, num_replicas, rank) {
    begin_index_ = rank_ * local_chunk_count_;
    end_index_ = begin_index_ + local_chunk_count_;
    chunk_index_ = begin_index_;
  }

  optional<size_t> next() {
    size_t idx = chunk_index_.fetch_add(1, std::memory_order_relaxed);
    if (idx < end_index_) {
      return idx % chunk_count_;
    } else {
      return nullopt;/// Select chunks randomly. The chunk order shuffled at each `reset()` call
    }
  }

  void reset() override {
    chunk_index_ = begin_index_;
  }

  private:
  size_t begin_index_;
  size_t end_index_;
  std::atomic<size_t> chunk_index_;
};
} // namespace detail

/// Options to configure a `ChunkDataset`.
struct ChunkDatasetOptions {
  ChunkDatasetOptions() = delete;
  ChunkDatasetOptions(
      size_t preloader_count,
      size_t batch_size,
      size_t cache_size = 2048)
      : preloader_count_(preloader_count),
        batch_size_(batch_size),
        cache_size_(cache_size) {
    AT_CHECK(
        preloader_count_ > 0,
        "Preloader count is 0. At least one preloader needs to be specified.");
    AT_CHECK(
        batch_size_ > 0,
        "Batch size is 0. A positive batch size needs to be specified.");
    AT_CHECK(
        cache_size_ > 0,
        "Cache size is 0. A positive cache size needs to be specified.");
    AT_CHECK(
        cache_size_ >= batch_size_,
        "Cache size is less than batch size. Cache needs to be large enough to "
        "hold at least one batch.");
  }

  /// The number of worker thread to preload chunk data.
  TORCH_ARG(size_t, preloader_count);

  /// The size of each batch.
  TORCH_ARG(size_t, batch_size);

  // The capacity of the queue for batch caching.
  TORCH_ARG(size_t, cache_size) = 2048;
};

/// A stateful dataset that support hierarchical sampling and prefetching of
/// entre chunks.
///
/// Unlike regular dataset, chunk dataset require two samplers to operate and
/// keeps an internal state. `ChunkSampler` selects, which chunk to load next,
/// while the `ExampleSampler` determins the order of Examples that are returned
/// in each `get_batch` call. The hierarchical sampling approach used here is
/// inspired by this paper http://martin.zinkevich.org/publications/nips2010.pdf
template <
    typename ChunkReader,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataset final
    : public StatefulDataset<
          ChunkDataset<ChunkReader, ExampleSampler>,
          typename ChunkReader::BatchType,
          size_t> {
 public:
  using BatchType = torch::optional<typename ChunkReader::BatchType>;
  using UnwrappedBatchType = typename ChunkReader::BatchType;
  using BatchRequestType = size_t;
  using ExampleSamplerType = ExampleSampler;

  ChunkDataset(
      ChunkReader chunk_reader,
      ExampleSampler example_sampler,
      std::shared_ptr<detail::ChunkSelector> chunk_selector,
      ChunkDatasetOptions options)
      : chunk_reader_(std::move(chunk_reader)),
        example_sampler_(std::move(example_sampler)),
        chunk_selector_(std::move(chunk_selector)),
        options_(std::move(options)),
        quit_worker_(false) {}

  virtual ~ChunkDataset() {
    free_workers();
  }

  /// Default get_batch method of BatchDataset. This method returns
  /// Example batches created from the preloaded chunks. The implemenation
  /// is dataset agnostic and does not need overriding in different chunk
  /// datasets.
  BatchType get_batch(size_t batch_size) override {
    AT_CHECK(
      batch_buffer_ != nullptr,
      "Dataset needs to call reset() before calling get_batch().");

    AT_CHECK(
      batch_size == options_.batch_size_,
      "The requested batch size does not match with the initialized batch size.\n"
      " The requested batch size is ", batch_size,
      ", while the dataset is created with batch size equal to ", options_.batch_size_);

    return batch_buffer_->get_batch();
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset.
  void reset() override {
    // free workers from previous reset if there is any.
    free_workers();
    preload_threads_.clear();
        
    chunk_reader_.reset();

    // reset the chunk selector.
    chunk_selector_->reset();

    // In distributed training, local chunk count could be different to total
    // chunks availble. Chunk selector holds the truth.
    size_t chunks_to_load = chunk_selector_->local_chunk_count();

    // Throw out any existing cached batch in the buffer and re-creates a new
    // chunk buffer.
    batch_buffer_ = torch::make_unique<
        detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>(
        chunks_to_load,
        options_.batch_size_,
        example_sampler_,
        options_.cache_size_);

    // create new workers for this new epoch.
    quit_worker_ = false;

    for (size_t i = 0; i < options_.preloader_count_; ++i) {
      preload_threads_.emplace_back(
          [this, i]() { this->preloader(i); });
    }
  }

  /// size is not used for chunk dataset.
  optional<size_t> size() const override {
    return torch::nullopt;
  }

 private:
  /// running on worker thread to preload chunk data.
  void preloader(size_t id) {
    while (!quit_worker_.load()) {
      try {
        size_t chunk_id = 0;
        if (auto chunk_sampler_result = chunk_selector_->next()) {
          chunk_id = chunk_sampler_result.value();
        } else {
          break;
        }
        UnwrappedBatchType data = chunk_reader_.read_chunk(chunk_id);
        if (data.empty()) {
          // if the chunk is empty, skip the current chunk data and move on to
          // the next.
          batch_buffer_->skip_chunk();
        }
        else {
          batch_buffer_->add_chunk_data(std::move(data));
        }
      } catch (...) {
        batch_buffer_->add_chunk_data(std::current_exception());
      }
    }
  }

  /// Block the current thread until the workers finish execution and exit.
  void free_workers() {
    if (!quit_worker_.load()) {
      quit_worker_ = true;
      if(batch_buffer_){
        batch_buffer_->stop();
      }
      for (auto& worker_thread : preload_threads_) {
        worker_thread.join();
      }
    }
  }

 private:
  // Templated class that defines what is a chunk and how to read chunk data.
  // When a chunk is returned by chunk_reader_, ChunkDataset split it into
  // batches and caches them in batch_buffer_.
  ChunkReader chunk_reader_;

  // example sampler to shuffle examples in a specific chunk
  ExampleSamplerType example_sampler_;

  // Selects chunks and their order for this reader.
  std::shared_ptr<detail::ChunkSelector> chunk_selector_;

  // batch data buffer which holds chunk data from preloading thread.
  std::shared_ptr<
      detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>
      batch_buffer_;

  // worker thread pool
  std::vector<std::thread> preload_threads_;

  /// The options the Dataset was configured with.
  const ChunkDatasetOptions options_;

  // indicate whether the worker thread can be teared down
  std::atomic<bool> quit_worker_;
};
} // namespace datasets
} // namespace data
} // namespace torch
