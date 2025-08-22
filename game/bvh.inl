namespace bot {

LeafID BVH::reserveLeaf(BodyID e, int32_t obj_id)
{
  int32_t leaf_idx = num_leaves_.fetch_add_relaxed(1);
  assert(leaf_idx < num_allocated_leaves_);

  leaf_entities_[leaf_idx] = e;
  leaf_obj_ids_[leaf_idx] = obj_id;

  return LeafID {
    leaf_idx,
  };
}

AABB BVH::getLeafAABB(LeafID leaf_id) const
{
  return leaf_aabbs_[leaf_id.id];
}

template <typename Fn>
void BVH::findIntersecting(const AABB &aabb, Fn &&fn) const
{
  int32_t stack[32];
  stack[0] = 0;
  int64_t stack_size = 1;

  while (stack_size > 0) {
    int32_t node_idx = stack[--stack_size];
    const Node &node = nodes_[node_idx];
    for (int i = 0; i < 4; i++) {
      if (!node.hasChild(i)) {
        continue; // Technically this could be break?
      };

      AABB child_aabb {
        /* .pMin = */ {
          node.minX[i],
          node.minY[i],
          node.minZ[i],
        },
          /* .pMax = */ {
            node.maxX[i],
            node.maxY[i],
            node.maxZ[i],
          },
      };

      if (aabb.overlaps(child_aabb)) {
        if (node.isLeaf(i)) {
          BodyID e = leaf_entities_[node.leafIDX(i)];
          fn(e);
        } else {
          stack[stack_size++] = node.children[i];
        }
      }
    }
  }
}

template <typename Fn>
void BVH::findLeafIntersecting(LeafID leaf_id, Fn &&fn) const
{
  AABB leaf_aabb = leaf_aabbs_[leaf_id.id];
  findIntersecting(leaf_aabb, std::forward<Fn>(fn));
}

void BVH::rebuildOnUpdate()
{
  force_rebuild_ = true;
}

void BVH::clearLeaves()
{
  num_leaves_.store_relaxed(0);
}

bool BVH::Node::isLeaf(int64_t child) const
{
  return children[child] & 0x80000000;
}

int32_t BVH::Node::leafIDX(int64_t child) const
{
  return children[child] & ~0x80000000;
}

void BVH::Node::setLeaf(int64_t child, int32_t idx)
{
  children[child] = 0x80000000 | idx;
}

void BVH::Node::setInternal(int64_t child, int32_t internal_idx)
{
  children[child] = internal_idx;
}

bool BVH::Node::hasChild(int64_t child) const
{
  return children[child] != sentinel_;
}

void BVH::Node::clearChild(int64_t child)
{
  children[child] = sentinel_;
}
  
}
