package btree

import "sync"

/*
* @CreateTime: 2021/2/25 15:01
* @Author: hujiaming
* @Description:
 */

// FreeList represents a free list of btree nodes. By default each
// BTree has its own FreeList, but multiple BTrees can share the same
// FreeList.
// Two Btrees using the same freelist are safe for concurrent write access.
// FreeList 当成一个 node 的 pool 使用。不同的 B+tree 可以并发读写这个 pool
type FreeList struct {
	mu       sync.Mutex
	freelist []*node
}

// NewFreeList creates a new free list.
// size is the maximum size of the returned free list.
func NewFreeList(size int) *FreeList {
	return &FreeList{freelist: make([]*node, 0, size)}
}

func (f *FreeList) newNode() (n *node) {
	f.mu.Lock()
	index := len(f.freelist) - 1 // 取最后一个
	if index < 0 {
		f.mu.Unlock()
		return new(node) // 如果 node slice 中没有元素，直接从内存申请
	}
	n = f.freelist[index]
	f.freelist[index] = nil // 置空，帮助 GC
	f.freelist = f.freelist[:index]
	f.mu.Unlock()
	return
}

// freeNode adds the given node to the list, returning true if it was added
// and false if it was discarded.
// freeNode 将一个 node 放回 pool 中。如果成功放回，返回 true
func (f *FreeList) freeNode(n *node) (out bool) {
	f.mu.Lock()
	if len(f.freelist) < cap(f.freelist) {
		f.freelist = append(f.freelist, n)
		out = true
	}
	f.mu.Unlock()
	return
}

// copyOnWriteContext pointers determine node ownership... a tree with a write
// context equivalent to a node's write context is allowed to modify that node.
// A tree whose write context does not match a node's is not allowed to modify
// it, and must create a new, writable copy (IE: it's a Clone).
//
// When doing any write operation, we maintain the invariant that the current
// node's context is equal to the context of the tree that requested the write.
// We do this by, before we descend into any node, creating a copy with the
// correct context if the contexts don't match.
//
// Since the node we're currently visiting on any write has the requesting
// tree's context, that node is modifiable in place.  Children of that node may
// not share context, but before we descend into them, we'll make a mutable
// copy.
type copyOnWriteContext struct {
	freelist *FreeList
}

func (c *copyOnWriteContext) newNode() (n *node) {
	n = c.freelist.newNode()
	n.cow = c
	return
}

// freeNode frees a node within a given COW context, if it's owned by that
// context.  It returns what happened to the node (see freeType const
// documentation).
// `Copy On Write` 技术理解：正常情况下，从父进程 folk 出一个子进程，父进程的资源也要复制一份给子进程；但是有时候子进程用不到，这个时候这种资源的
// 复制就是一种浪费。于是 COW 优化策略出现了：
// copy 的时候，将父进程的资源设置为 read-only，父进程和子进程共享这个资源；
// 当有一个进程发生 write 操作时，会触发中断：将这份资源复制一份，子进程与父进程各自执有一份，互不干扰。
func (c *copyOnWriteContext) freeNode(n *node) freeType {
	// 其实是通过 node.cow 来判断当前的 context 属于哪个 b+tree
	if n.cow == c {
		// clear to allow GC
		n.items.truncate(0)    // 将这个 node 的所有 items(索引) 清空
		n.children.truncate(0) // 将这个 node 的所有孩子节点清空
		n.cow = nil
		// 将这个 node 放回 pool
		if c.freelist.freeNode(n) {
			return ftStored
		} else {
			return ftFreelistFull
		}
	} else {
		return ftNotOwned
	}
}
