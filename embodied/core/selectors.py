import collections
import threading

import numpy as np


class UncertaintyCaching:
  """
  Uncertainty caching sampling. Does weighted sampling based on
  uncertainty values. If no uncertainty is provided, it defaults to
  sampling the first item in the list. Uncertainties are recomputed every N steps.

  Args:
    seed: Random seed for sampling.
    weighted_sampling: If True, samples based on uncertainty values.
      If False, samples the first item in the list.
  """

  def __init__(self, seed=0, weighted_sampling=True):
    self.itemids = []
    self.lock = threading.Lock()
    self.rng = np.random.default_rng(seed)
    self.weighted_sampling = weighted_sampling
    self.uncertainty = {}
    self.mean = 1.0
    self.std = 0.0

  def __len__(self):
    return len(self.itemids)

  def __call__(self, batch_size=1, mode='report'):
    """
    Sample an item based on uncertainty values. 

    Args:
      batch_size: Number of items to sample.
      mode: 'train' for training mode, 'report' for reporting mode.
    Returns:
      A list of sampled itemids. If batch_size is 1, returns a single itemid.
    """
    with self.lock:
      itemids = list(self.itemids)
    if not itemids:
      raise ValueError("No itemids to sample from.")
    
    if mode == 'train':
      # Get the uncertainty values for the itemids.
      values = np.array([self.uncertainty[itemid] for itemid in itemids])

      # Weighted sampling based on uncertainty values.
      if self.weighted_sampling:
        probs = values / values.sum() if values.sum() > 0 else np.ones_like(values) / len(values)
        assert np.isclose(probs.sum(), 1.0), f"Probabilities do not sum to 1: {probs.sum()}"
        idx = self.rng.choice(len(itemids), size=batch_size, p=probs)

      # Take the top-k items based on uncertainty values.
      else:
        idx = np.argsort(values)[-batch_size:]

    # Uniform sampling for reporting
    elif mode == 'report':
      replace = batch_size > len(self.itemids)
      idx = self.rng.choice(len(itemids), size=batch_size, replace=replace)

    return [self.itemids[id] for id in idx] if len(idx) > 1 else self.itemids[idx[0]]

  def __setitem__(self, itemid, stepids, uncertainty=None):
    with self.lock:
      if itemid not in self.itemids:
        self.itemids.append(itemid)
      if uncertainty is not None: 
        self.uncertainty[itemid] = uncertainty
      else: 
       self.uncertainty[itemid] = self.mean + self.std

  def __delitem__(self, itemid):
    with self.lock:
      if itemid in self.uncertainty:
        del self.uncertainty[itemid]
      if itemid in self.itemids:
        self.itemids.remove(itemid)

  def list_items(self):
    with self.lock:
      return list(self.itemids)
    
  def update_uncertainty(self, uncertainties, itemids):
    with self.lock:
      for itemid, uncertainty in uncertainties.items():
        if uncertainty is not None:
          self.uncertainty[itemid] = uncertainty
      if self.itemids:
        self.mean = np.mean(list(self.uncertainty.values()))
        self.std = np.std(list(self.uncertainty.values()))


class Fifo:

  def __init__(self):
    self.queue = collections.deque()

  def __call__(self):
    return self.queue[0]

  def __len__(self):
    return len(self.queue)

  def __setitem__(self, key, stepids):
    self.queue.append(key)

  def __delitem__(self, key):
    if self.queue[0] == key:
      self.queue.popleft()
    else:
      # This is very slow but typically not used.
      self.queue.remove(key)


class Uniform:

  def __init__(self, seed=0):
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

  def __len__(self):
    return len(self.keys)

  def __call__(self):
    with self.lock:
      index = self.rng.integers(0, len(self.keys)).item()
      return self.keys[index]

  def __setitem__(self, key, stepids):
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)

  def __delitem__(self, key):
    with self.lock:
      assert 2 <= len(self), len(self)
      index = self.indices.pop(key)
      last = self.keys.pop()
      if index != len(self.keys):
        self.keys[index] = last
        self.indices[last] = index


class Recency:

  def __init__(self, uprobs, seed=0):
    assert uprobs[0] >= uprobs[-1], uprobs
    self.uprobs = uprobs
    self.tree = self._build(uprobs)
    self.rng = np.random.default_rng(seed)
    self.step = 0
    self.steps = {}
    self.items = {}

  def __len__(self):
    return len(self.items)

  def __call__(self):
    for retry in range(10):
      try:
        age = self._sample(self.tree, self.rng)
        if len(self.items) < len(self.uprobs):
          age = int(age / len(self.uprobs) * len(self.items))
        return self.items[self.step - 1 - age]
      except KeyError:
        # Item might have been deleted very recently.
        if retry < 9:
          import time
          time.sleep(0.01)
        else:
          raise

  def __setitem__(self, key, stepids):
    self.steps[key] = self.step
    self.items[self.step] = key
    self.step += 1

  def __delitem__(self, key):
    step = self.steps.pop(key)
    del self.items[step]

  def _sample(self, tree, rng, bfactor=16):
    path = []
    for level, prob in enumerate(tree):
      p = prob
      for segment in path:
        p = p[segment]
      index = rng.choice(len(p), p=p)
      path.append(index)
    index = sum(
        index * bfactor ** (len(tree) - level - 1)
        for level, index in enumerate(path))
    return index

  def _build(self, uprobs, bfactor=16):
    assert np.isfinite(uprobs).all(), uprobs
    assert (uprobs >= 0).all(), uprobs
    depth = int(np.ceil(np.log(len(uprobs)) / np.log(bfactor)))
    size = bfactor ** depth
    uprobs = np.concatenate([uprobs, np.zeros(size - len(uprobs))])
    tree = [uprobs]
    for level in reversed(range(depth - 1)):
      tree.insert(0, tree[0].reshape((-1, bfactor)).sum(-1))
    for level, prob in enumerate(tree):
      prob = prob.reshape([bfactor] * (1 + level))
      total = prob.sum(-1, keepdims=True)
      with np.errstate(divide='ignore', invalid='ignore'):
        tree[level] = np.where(total, prob / total, prob)
    return tree


class Prioritized:
  """
  Prioritized sampling selector. Uses a sample tree to sample items
  based on their priority values. The priority values can be updated
  dynamically. The tree is built using a branching factor, which
  determines how many children each node can have. The tree is sampled
  using a random number generator, which can be seeded for reproducibility.

  Args:
    exponent: Exponent for the priority values. If 1.0, uses the mean
      of the priority values. If 0.0, uses the maximum priority value.
    initial: Initial priority value for new items. Can be mean+std or inf.
    zero_on_sample: If True, sets the priority of sampled items to zero.
    maxfrac: Maximum fraction of the maximum priority value to use in
      the aggregated priority value.
    branching: Branching factor for the sample tree.
    seed: Random seed for sampling.
  """

  def __init__(
      self, exponent=1.0, initial=1.0, zero_on_sample=False,
      maxfrac=0.0, branching=16, seed=0):
    assert 0 <= maxfrac <= 1, maxfrac
    self.exponent = float(exponent)
    self.zero_on_sample = zero_on_sample
    self.maxfrac = maxfrac
    self.tree = SampleTree(branching, seed)
    if initial == "mean+std":
      self.initial = 1.0
      self.prios = collections.defaultdict(lambda: self.mean + self.std)
    elif initial == "inf":
      self.initial = float('inf')
      self.prios = collections.defaultdict(lambda: self.initial)
    self.stepitems = collections.defaultdict(list)
    self.items = {}
    self.mean = self.initial
    self.std = 0.0

  def get_stats(self):
    mean, std = 1.0, 0.0
    if self.prios:
      prios = [p for p in self.prios.values() if np.isfinite(p)]
      mean = np.mean(prios)
      std = np.std(prios)
    self.mean = mean
    self.std = std
    return mean, std

  def prioritize(self, stepids, priorities):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    for stepid, priority in zip(stepids, priorities):
      try:
        self.prios[stepid] = priority
      except KeyError:
        print('Ignoring priority update for removed time step.')
    items = []
    for stepid in stepids:
      items += self.stepitems[stepid]
    for key in list(set(items)):
      try:
        self.tree.update(key, self._aggregate(key))
      except KeyError:
        print('Ignoring tree update for removed time step.')

  def __len__(self):
    return len(self.items)

  def __call__(self):
    key = self.tree.sample()
    if self.zero_on_sample:
      zeros = [0.0] * len(self.items[key])
      self.prioritize(self.items[key], zeros)
    return key

  def __setitem__(self, key, stepids):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    self.items[key] = stepids
    [self.stepitems[stepid].append(key) for stepid in stepids]
    self.tree.insert(key, self._aggregate(key))

  def __delitem__(self, key):
    self.tree.remove(key)
    stepids = self.items.pop(key)
    for stepid in stepids:
      stepitems = self.stepitems[stepid]
      stepitems.remove(key)
      if not stepitems:
        del self.stepitems[stepid]
        del self.prios[stepid]

  def _aggregate(self, key):
    # Both list comprehensions in this function are a performance bottleneck
    # because they are called very often.
    prios = [self.prios[stepid] for stepid in self.items[key]]
    if self.exponent != 1.0:
      prios = [x ** self.exponent for x in prios]
    mean = sum(prios) / len(prios)
    if self.maxfrac:
      return self.maxfrac * max(prios) + (1 - self.maxfrac) * mean
    else:
      return mean


class Mixture:

  def __init__(self, selectors, fractions, seed=0):
    assert set(selectors.keys()) == set(fractions.keys())
    assert sum(fractions.values()) == 1, fractions
    for key, frac in list(fractions.items()):
      if not frac:
        selectors.pop(key)
        fractions.pop(key)
    keys = sorted(selectors.keys())
    self.selectors = [selectors[key] for key in keys]
    self.fractions = np.array([fractions[key] for key in keys], np.float32)
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    return self.rng.choice(self.selectors, p=self.fractions)()

  def __setitem__(self, key, stepids):
    for selector in self.selectors:
      selector[key] = stepids

  def __delitem__(self, key):
    for selector in self.selectors:
      del selector[key]

  def prioritize(self, stepids, priorities):
    for selector in self.selectors:
      if hasattr(selector, 'prioritize'):
        selector.prioritize(stepids, priorities)


class SampleTree:

  def __init__(self, branching=16, seed=0):
    assert 2 <= branching
    self.branching = branching
    self.root = SampleTreeNode()
    self.last = None
    self.entries = {}
    self.rng = np.random.default_rng(seed)

  def __len__(self):
    return len(self.entries)

  def insert(self, key, uprob):
    if not self.last:
      node = self.root
    else:
      ups = 0
      node = self.last.parent
      while node and len(node) >= self.branching:
        node = node.parent
        ups += 1
      if not node:
        node = SampleTreeNode()
        node.append(self.root)
        self.root = node
      for _ in range(ups):
        below = SampleTreeNode()
        node.append(below)
        node = below
    entry = SampleTreeEntry(key, uprob)
    node.append(entry)
    self.entries[key] = entry
    self.last = entry

  def remove(self, key):
    entry = self.entries.pop(key)
    entry_parent = entry.parent
    last_parent = self.last.parent
    entry.parent.remove(entry)
    if entry is not self.last:
      entry_parent.append(self.last)
    node = last_parent
    ups = 0
    while node.parent and not len(node):
      above = node.parent
      above.remove(node)
      node = above
      ups += 1
    if not len(node):
      self.last = None
      return
    while isinstance(node, SampleTreeNode):
      node = node.children[-1]
    self.last = node

  def update(self, key, uprob):
    entry = self.entries[key]
    entry.uprob = uprob
    entry.parent.recompute()

  def sample(self):
    node = self.root
    while isinstance(node, SampleTreeNode):
      uprobs = np.array([x.uprob for x in node.children])
      total = uprobs.sum()
      if not np.isfinite(total):
        finite = np.isinf(uprobs)
        probs = finite / finite.sum()
      elif total == 0:
        probs = np.ones(len(uprobs)) / len(uprobs)
      else:
        probs = uprobs / total
      choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
      node = node.children[choice.item()]
    return node.key


class SampleTreeNode:

  __slots__ = ('parent', 'children', 'uprob')

  def __init__(self, parent=None):
    self.parent = parent
    self.children = []
    self.uprob = 0

  def __repr__(self):
    return (
        f'SampleTreeNode(uprob={self.uprob}, '
        f'children={[x.uprob for x in self.children]})'
    )

  def __len__(self):
    return len(self.children)

  def __bool__(self):
    return True

  def append(self, child):
    if child.parent:
      child.parent.remove(child)
    child.parent = self
    self.children.append(child)
    self.recompute()

  def remove(self, child):
    child.parent = None
    self.children.remove(child)
    self.recompute()

  def recompute(self):
    self.uprob = sum(x.uprob for x in self.children)
    self.parent and self.parent.recompute()


class SampleTreeEntry:

  __slots__ = ('parent', 'key', 'uprob')

  def __init__(self, key=None, uprob=None):
    self.parent = None
    self.key = key
    self.uprob = uprob