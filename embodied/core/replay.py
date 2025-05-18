import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind

import elements
import numpy as np
import jax.numpy as jnp
import jax
import ninjax as nj

from . import chunk as chunklib
from . import limiters
from . import selectors


class Replay:

  def __init__(
      self, length, capacity=None, directory=None, chunksize=1024,
      online=False, selector=None, save_wait=False, name='unnamed', seed=0, pred_next=None):

    self.length = length
    self.capacity = capacity
    self.chunksize = chunksize
    self.name = name

    self.sampler = selector if selector is not None else selectors.Uniform(seed)

    self._pure_pred_next = nj.pure(pred_next, nested=True)
    self.seed = jax.random.PRNGKey(jax.device_put(seed))

    self.chunks = {}
    self.refs = {}
    self.refs_lock = threading.RLock()

    self.items = {}
    self.fifo = deque()
    self.itemid = 0

    self.current = {}
    self.streams = defaultdict(deque)
    self.rwlock = elements.RWLock()

    self.online = online
    if online:
      self.lengths = defaultdict(int)
      self.queue = deque()

    if directory:
      self.directory = elements.Path(directory)
      self.directory.mkdir()
      self.workers = ThreadPoolExecutor(16, 'replay_saver')
      self.saved = set()
    else:
      self.directory = None
    self.save_wait = save_wait

    self.metrics = {'samples': 0, 'inserts': 0, 'updates': 0}

  def __len__(self):
    return len(self.items)

  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    chunk_nbytes = sum(x.nbytes for x in list(self.chunks.values()))
    stats = {
        'items': len(self.items),
        'chunks': len(self.chunks),
        'streams': len(self.streams),
        'ram_gb': chunk_nbytes / (1024 ** 3),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'updates': m['updates'],
        'replay_ratio': ratio(self.length * m['samples'], m['inserts']),
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  @elements.timer.section('replay_add')
  def add(self, step, worker=0, agent=None):
    step = {k: v for k, v in step.items() if not k.startswith('log/')}
    with self.rwlock.reading:
      step = {k: np.asarray(v) for k, v in step.items()}

      if worker not in self.current:
        chunk = chunklib.Chunk(self.chunksize)
        with self.refs_lock:
          self.refs[chunk.uuid] = 1
        self.chunks[chunk.uuid] = chunk
        self.current[worker] = (chunk.uuid, 0)

      chunkid, index = self.current[worker]
      step['stepid'] = np.frombuffer(
          bytes(chunkid) + index.to_bytes(4, 'big'), np.uint8)
      stream = self.streams[worker]
      chunk = self.chunks[chunkid]
      assert chunk.length == index, (chunk.length, index)
      chunk.append(step)
      assert chunk.length == index + 1, (chunk.length, index + 1)
      stream.append((chunkid, index))
      with self.refs_lock:
        self.refs[chunkid] += 1

      index += 1
      if index < chunk.size:
        self.current[worker] = (chunkid, index)
      else:
        self._complete(chunk, worker)
      assert len(self.streams) == len(self.current)

      if len(stream) >= self.length:
        # Increment is not thread safe thus inaccurate but faster than locking.
        self.metrics['inserts'] += 1
        chunkid, index = stream.popleft()
        self._insert(chunkid, index)

        if self.online and self.lengths[worker] % self.length == 0:
          self.queue.append((chunkid, index))

      if self.online:
        self.lengths[worker] += 1

  @elements.timer.section('replay_sample')
  def sample(self, batch, mode='train', agent=None):
    message = f'Replay buffer {self.name} is empty'
    limiters.wait(lambda: len(self.sampler), message)

    # Uncertainty sampling
    if agent is not None:
      # Uncertainty calulation and sampling
      uncertainty, itemids = self.calc_uncertainty(agent)
      itemids = self.sampler(uncertainty, batch_size=batch, itemids=itemids)

      # Gather sequences from itemids
      seqs, is_online = [], []
      for itemid in itemids:
        chunkid, index = self.items[itemid]
        seq = self._getseq(chunkid, index, concat=False)
        seqs.append(seq)
        is_online.append(False)

    # Other sampling methods
    else:
      seqs, is_online = zip(*[self._sample(mode) for _ in range(batch)])

    data = self._assemble_batch(seqs, 0, self.length)
    data = self._annotate_batch(data, is_online, True)
    return data
  
  def calc_uncertainty(self, agent=None):
    """
    Vectorized uncertainty calculation for each sequence in the replay buffer.
    For each step in the sequence, we predict the stochastic representation of the next 
    timestep and calculate the KL divergence.
    The uncertainty is the average KL divergence over all steps in the sequence.

    Args:
      agent: The agent used to predict the next timestep.

    Returns:
      uncertainty: A dictionary mapping item IDs to their corresponding uncertainty values.
      itemids: A list of item IDs. For threading safety, this is a copy of the item IDs in the sampler.
    """
    # uncertainty = {}
    # itemids = self.sampler.list_items()
    # for itemid in itemids:
    #   chunkid, index = self.items[itemid]
    #   sequence = self._getseq(chunkid, index, concat=True)
      
    #   # Calculate average uncertainty for the sequence
    #   mean_uncertainty = 0
    #   for idx in range(len(sequence['dyn/stoch']) - 1):
    #     current_timestep = {
    #       'dyn/deter': sequence['dyn/deter'][idx],
    #       'dyn/stoch': sequence['dyn/stoch'][idx],\
    #       'action': sequence['action'][idx],
    #     }
    #     true_next_stoch = sequence['dyn/stoch'][idx+1]

    #     # Predict the next timestep based on the current timestep
    #     state, pred_next_stoch = self._pure_pred_next({}, current_timestep, seed=self.seed, create=True)

    #     # Calculate the uncertainty
    #     pred_distr = agent.model.dyn._dist(pred_next_stoch)
    #     true_distr = agent.model.dyn._dist(true_next_stoch)
    #     mean_uncertainty += pred_distr.kl(true_distr)[0]
      
    #   mean_uncertainty /= len(sequence['dyn/stoch']) - 1
    #   uncertainty[itemid] = mean_uncertainty
      
    # return uncertainty, itemids

    # VECTORIZED VERSION
    # This version is more efficient but requires a lot of memory.
    uncertainty = {}
    # Gather all sequences to process in a batch
    itemids = self.sampler.list_items()
    batch_sequences = []
    for itemid in itemids:
      chunkid, index = self.items[itemid]
      sequence = self._getseq(chunkid, index, concat=True)
      batch_sequences.append(sequence)

    # Find the maximum sequence length for batching
    max_seq_len = max(len(seq['dyn/stoch']) for seq in batch_sequences) - 1
    batch_size = len(batch_sequences)

    # Pad or truncate sequences to max_seq_len
    def pad_to(seq, key, pad_value=0):
      arr = seq[key]
      if arr.shape[0] >= max_seq_len + 1:
        return arr[:max_seq_len + 1]
      pad_width = [(0, max_seq_len + 1 - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
      return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)

    padded_sequences = []
    for seq in batch_sequences:
      padded_seq = {
        'dyn/deter': pad_to(seq, 'dyn/deter'),
        'dyn/stoch': pad_to(seq, 'dyn/stoch'),
        'action': pad_to(seq, 'action')
      }
      padded_sequences.append(padded_seq)
    batch_sequences = padded_sequences
    seq_len = max_seq_len

    # Prepare batched arrays
    deter = np.stack([seq['dyn/deter'][:seq_len:2] for seq in batch_sequences])  # (B, T, ...) 
    stoch = np.stack([seq['dyn/stoch'][:seq_len:2] for seq in batch_sequences])
    actions = np.stack([seq['action'][:seq_len:2] for seq in batch_sequences])
    true_next_stoch = np.stack([seq['dyn/stoch'][1:seq_len+1:2] for seq in batch_sequences])

    # Prepare input for pred_next
    def pred_next_batch(deter, stoch, action):
      # deter, stoch, action: (B, T, ...)
      def single_step(d, s, a):
        timestep = {
          'dyn/deter': d,
          'dyn/stoch': s,
          'action': a,
        }
        # state is unused, pass {}
        _, pred = self._pure_pred_next({}, timestep, seed=self.seed, create=True)
        return pred
      # Vectorize over batch and time
      return jax.vmap(jax.vmap(single_step, in_axes=(0,0,0)), in_axes=(0,0,0))(deter, stoch, actions)

    pred_next_stoch = pred_next_batch(deter, stoch, actions)  # (B, T, ...)

    def kl_fn(pred_stoch, true_stoch):
      pred_dist = agent.model.dyn._dist(pred_stoch)
      true_dist = agent.model.dyn._dist(true_stoch)
      return pred_dist.kl(true_dist)[0]  # or whatever shape you want

    # Now vmap over batch and time
    kls = jax.vmap(jax.vmap(kl_fn))(pred_next_stoch, true_next_stoch)  # (B, T)

    mean_uncertainty = np.array(kls).mean(axis=1)  # (B,)

    for i, itemid in enumerate(itemids):
      uncertainty[itemid] = mean_uncertainty[i]

    return uncertainty, itemids
  
  @elements.timer.section('replay_update')
  def update(self, data):
    stepid = data.pop('stepid')
    priority = data.pop('priority', None)
    assert stepid.ndim == 3, stepid.shape
    self.metrics['updates'] += int(np.prod(stepid.shape[:-1]))

    if priority is not None:
      assert priority.ndim == 2, priority.shape
      self.sampler.prioritize(
          stepid.reshape((-1, stepid.shape[-1])),
          priority.flatten())
       
    if data:
      for i, stepid in enumerate(stepid):
        stepid = stepid[0].tobytes()
        chunkid = elements.UUID(stepid[:-4])
        index = int.from_bytes(stepid[-4:], 'big')
        values = {k: v[i] for k, v in data.items()}
        try:
          self._setseq(chunkid, index, values)
        except KeyError:
          pass

  def _sample(self, mode, uncertainty=None):
    assert mode in ('train', 'report', 'eval'), mode
    if mode == 'train':
      # Increment is not thread safe thus inaccurate but faster than locking.
      self.metrics['samples'] += 1
    while True:
      try:
        if self.online and self.queue and mode == 'train':
          chunkid, index = self.queue.popleft()
          is_online = True
        else:
          with elements.timer.section('sample'):
            if uncertainty is not None:
              itemid = self.sampler(uncertainty)
            else:
              itemid = self.sampler()
          chunkid, index = self.items[itemid]
          is_online = False
        seq = self._getseq(chunkid, index, concat=False)
        return seq, is_online
      except KeyError:
        continue

  def _insert(self, chunkid, index):
    while self.capacity and len(self.items) >= self.capacity:
      self._remove()
    itemid = self.itemid
    self.itemid += 1
    self.items[itemid] = (chunkid, index)
    stepids = self._getseq(chunkid, index, ['stepid'])['stepid']
    self.sampler[itemid] = stepids
    self.fifo.append(itemid)

  def _remove(self):
    itemid = self.fifo.popleft()
    del self.sampler[itemid]
    chunkid, index = self.items.pop(itemid)
    with self.refs_lock:
      self.refs[chunkid] -= 1
      if self.refs[chunkid] < 1:
        del self.refs[chunkid]
        chunk = self.chunks.pop(chunkid)
        if chunk.succ in self.refs:
          self.refs[chunk.succ] -= 1

  def _getseq(self, chunkid, index, keys=None, concat=True):
    chunk = self.chunks[chunkid]
    available = chunk.length - index
    if available >= self.length:
      with elements.timer.section('get_slice'):
        seq = chunk.slice(index, self.length)
        if not concat:
          seq = {k: [v] for k, v in seq.items()}
        return seq
    else:
      with elements.timer.section('get_compose'):
        parts = [chunk.slice(index, available)]
        remaining = self.length - available
        while remaining > 0:
          chunk = self.chunks[chunk.succ]
          used = min(remaining, chunk.length)
          parts.append(chunk.slice(0, used))
          remaining -= used
        seq = {k: [p[k] for p in parts] for k in keys or parts[0].keys()}
        if concat:
          seq = {k: np.concatenate(v, 0) for k, v in seq.items()}
        return seq

  def _setseq(self, chunkid, index, values):
    length = len(next(iter(values.values())))
    chunk = self.chunks[chunkid]
    available = chunk.length - index
    if available >= length:
      with elements.timer.section('set_slice'):
        return chunk.update(index, length, values)
    else:
      with elements.timer.section('set_compose'):
        part = {k: v[:available] for k, v in values.items()}
        values = {k: v[available:] for k, v in values.items()}
        chunk.update(index, available, part)
        remaining = length - available
        while remaining > 0:
          chunk = self.chunks[chunk.succ]
          used = min(remaining, chunk.length)
          part = {k: v[:used] for k, v in values.items()}
          values = {k: v[used:] for k, v in values.items()}
          chunk.update(0, used, part)
          remaining -= used

  # def dataset(self, batch, length=None, consec=None, prefix=0, report=False):
  #   length = length or self.length
  #   consec = consec or (self.length - prefix) // length
  #   assert consec <= (self.length - prefix) // length, (
  #       self.length, length, consec, prefix)
  #   limiters.wait(lambda: len(self.sampler), 'Replay buffer is empty')
  #   # For performance, each batch should be consecutive in memory, rather than
  #   # a non-consecutive view into a longer batch. For example, this allows
  #   # near-instant serialization when sending over the network.
  #   while True:
  #     seqs, is_online = zip(*[self._sample(report) for _ in range(batch)])
  #     for i in range(consec):
  #       offset = i * length
  #       data = self._assemble_batch(seqs, offset, offset + length + prefix)
  #       data = self._annotate_batch(data, is_online, is_first=(i == 0))
  #       data['consec'] = np.full(data['is_first'].shape, i, np.int32)
  #       yield data

  @elements.timer.section('assemble_batch')
  def _assemble_batch(self, seqs, start, stop):
    shape = (len(seqs), stop - start)
    data = {
        key: np.empty((*shape, *parts[0].shape[1:]), parts[0].dtype)
        for key, parts in seqs[0].items()}
    for n, seq in enumerate(seqs):
      st, dt = 0, 0  # Source and destination time index.
      for p in range(len(seq['stepid'])):
        partlen = len(seq['stepid'][p])
        if start < st + partlen:
          part_start = max(0, start - st)
          part_stop = min(stop - st, partlen)
          num = part_stop - part_start
          for k in data.keys():
            data[k][n, dt: dt + num] = seq[k][p][part_start: part_stop]
          dt += num
        st += partlen
        if st >= stop:
          break
    return data

  @elements.timer.section('annotate_batch')
  def _annotate_batch(self, data, is_online, is_first):
    data = data.copy()
    # if self.online:
    #   broadcasted = [[x] for x in is_online]
    #   data['is_online'] = np.full(data['is_first'].shape, broadcasted, bool)
    if 'is_first' in data:
      if is_first:
        data['is_first'] = data['is_first'].copy()
        data['is_first'][:, 0] = True
      if 'is_last' in data:
        # Make sure that abandoned episodes have is_last set.
        next_is_first = np.roll(data['is_first'], shift=-1, axis=1)
        next_is_first[:, -1] = False
        data['is_last'] = data['is_last'] | next_is_first
    return data

  @elements.timer.section('replay_save')
  def save(self):
    if self.directory:
      with self.rwlock.writing:
        for worker, (chunkid, _) in self.current.items():
          chunk = self.chunks[chunkid]
          if chunk.length > 0:
            self._complete(chunk, worker)
        promises = []
        for chunk in self.chunks.values():
          if chunk.length > 0 and chunk.uuid not in self.saved:
            self.saved.add(chunk.uuid)
            promises.append(self.workers.submit(chunk.save, self.directory))
        if self.save_wait:
          [promise.result() for promise in promises]
    return None

  @elements.timer.section('replay_load')
  def load(self, data=None, directory=None, amount=None):

    directory = directory or self.directory
    amount = amount or self.capacity or np.inf
    if not directory:
      return
    revsorted = lambda x: list(reversed(sorted(list(x))))
    directory = elements.Path(directory)
    names_loaded = revsorted(x.filename for x in list(self.chunks.values()))
    names_ondisk = revsorted(x.name for x in directory.glob('*.npz'))
    names_ondisk = [x for x in names_ondisk if x not in names_loaded]
    if not names_ondisk:
      return

    numitems = self._numitems(names_loaded + names_ondisk)
    uuids = [elements.UUID(x.split('-')[1]) for x in names_ondisk]
    total = 0
    numchunks = 0
    for uuid in uuids:
      numchunks += 1
      total += numitems[uuid]
      if total >= amount:
        break

    load = bind(chunklib.Chunk.load, error='none')
    filenames = [directory / x for x in names_ondisk[:numchunks]]

    with ThreadPoolExecutor(16, 'replay_loader') as pool:
      chunks = [x for x in pool.map(load, filenames) if x]

    # We need to recompute the number of items per chunk now because some
    # chunks may be corrupted and thus not available.
    # numitems = self._numitems(chunks + list(self.chunks.values()))
    numitems = self._numitems(chunks)

    with self.rwlock.writing:
      self.saved.update([chunk.uuid for chunk in chunks])
      with self.refs_lock:
        for chunk in chunks:
          self.chunks[chunk.uuid] = chunk
          self.refs[chunk.uuid] = 0
        for chunk in reversed(chunks):
          amount = numitems[chunk.uuid]
          self.refs[chunk.uuid] += amount
          if chunk.succ in self.refs:
            self.refs[chunk.succ] += 1
          for index in range(amount):
            self._insert(chunk.uuid, index)

  @elements.timer.section('complete_chunk')
  def _complete(self, chunk, worker):
    succ = chunklib.Chunk(self.chunksize)
    with self.refs_lock:
      self.refs[chunk.uuid] -= 1
      self.refs[succ.uuid] = 2
    self.chunks[succ.uuid] = succ
    self.current[worker] = (succ.uuid, 0)
    chunk.succ = succ.uuid
    return succ

  def _numitems(self, chunks):
    chunks = [x.filename if hasattr(x, 'filename') else x for x in chunks]
    if not chunks:
      return 0
    chunks = list(reversed(sorted([elements.Path(x).stem for x in chunks])))
    times, uuids, succs, lengths = zip(*[x.split('-') for x in chunks])
    uuids = [elements.UUID(x) for x in uuids]
    succs = [elements.UUID(x) for x in succs]
    lengths = {k: int(v) for k, v in zip(uuids, lengths)}
    future = {}
    for uuid, succ in zip(uuids, succs):
      future[uuid] = lengths[uuid] + future.get(succ, 0)
    numitems = {}
    for uuid, succ in zip(uuids, succs):
      numitems[uuid] = lengths[uuid] + 1 - self.length + future.get(succ, 0)
    numitems = {k: np.clip(v, 0, lengths[k]) for k, v in numitems.items()}
    return numitems

  def _notempty(self, reason=False):
    if reason:
      return (True, 'ok') if len(self.sampler) else (False, 'empty buffer')
    else:
      return bool(len(self.sampler))
