/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package s2

import (
	"container/heap"
)

// RegionCoverer allows arbitrary regions to be approximated as unions of cells (CellUnion).
// This is useful for implementing various sorts of search and precomputation operations.
//
// Typical usage:
//
//	rc := &s2.RegionCoverer{MaxLevel: 30, MaxCells: 5}
//	r := s2.Region(CapFromCenterArea(center, area))
//	covering := rc.Covering(r)
//
// This yields a CellUnion of at most 5 cells that is guaranteed to cover the
// given region (a disc-shaped region on the sphere).
// 覆盖的区域是球面上的盘子区域
//
// For covering, only cells where (level - MinLevel) is a multiple of LevelMod will be used.
// This effectively allows the branching factor of the S2 CellID hierarchy to be increased.
// Currently the only parameter values allowed are 0/1, 2, or 3, corresponding to
// branching factors of 4, 16, and 64 respectively.
//
// Note the following:
//
//  - MinLevel takes priority over MaxCells, i.e. cells below the given level will
//    never be used even if this causes a large number of cells to be returned.
//		低等级的 level 的 cell 会优先使用。即大格子会优先使用。

//  - For any setting of MaxCells, up to 6 cells may be returned if that
//    is the minimum number of cells required (e.g. if the region intersects
//    all six face cells).  Up to 3 cells may be returned even for very tiny
//    convex regions if they happen to be located at the intersection of
//    three cube faces.
//
// 		关于 MaxCells 的设置，如果是所需的最小的单元数，就返回最小的单元数。如果待覆盖的区域正好位于三个立方体的交点处，那么就要返回3个 cell，即使覆盖的面会比要求的大一些。
//
//  - For any setting of MaxCells, an arbitrary number of cells may be
//    returned if MinLevel is too high for the region being approximated.
//
//		如果设置的单元格的最小 cell 的 level 太高了，即格子太小了，那么就会返回任意数量的单元格数量。
//
//  - If MaxCells is less than 4, the area of the covering may be
//    arbitrarily large compared to the area of the original region even if
//    the region is convex (e.g. a Cap or Rect).
//
//		如果 MaxCells 小于4，即使该区域是凸的，比如 cap 或者 rect ，最终覆盖的面积也要比原生区域大。
//
// The approximation algorithm is not optimal but does a pretty good job in
// practice. The output does not always use the maximum number of cells
// allowed, both because this would not always yield a better approximation,
// and because MaxCells is a limit on how much work is done exploring the
// possible covering as well as a limit on the final output size.
//
//
//	这个近似算法并不是最优算法，但是在实践中效果还不错。输出的结果并不总是使用的满足条件的最多的单元数，
//  因为这样也不是总能产生更好的近似结果(比如上面举例的，区域整好位于三个面的交点处，得到的结果比原区域要大很多)
//	并且 MaxCells 对搜索的工作量和最终输出的 cell 的数量是一种限制。
//
//
// Because it is an approximation algorithm, one should not rely on the
// stability of the output. In particular, the output of the covering algorithm
// may change across different versions of the library.
//
// One can also generate interior coverings, which are sets of cells which
// are entirely contained within a region. Interior coverings can be
// empty, even for non-empty regions, if there are no cells that satisfy
// the provided constraints and are contained by the region. Note that for
// performance reasons, it is wise to specify a MaxLevel when computing
// interior coverings - otherwise for regions with small or zero area, the
// algorithm may spend a lot of time subdividing cells all the way to leaf
// level to try to find contained cells.

// 由于这是一个近似算法，所以不能依赖它输出的稳定性。特别的，覆盖算法的输出结果会在不同的库的版本上有所不同。
//
// 这个算法还可以产生内部覆盖的 cell，内部覆盖的 cell 指的是完全被包含在区域内的 cell。
// 如果没有满足条件的 cell ，即使对于非空区域，内部覆盖 cell 也可能是空的。
// 请注意，处于性能考虑，在计算内部覆盖 cell 的时候，指定 MaxLevel 是明智的做法。
// 否则，对于小的或者零面积的区域，算法可能会花费大量时间将单元格细分到叶子 level ，以尝试找到满足条件的内部覆盖单元格 cell。

type RegionCoverer struct {
	MinLevel int // the minimum cell level to be used.
	MaxLevel int // the maximum cell level to be used.
	LevelMod int // the LevelMod to be used.
	MaxCells int // the maximum desired number of cells in the approximation.
}

type coverer struct {
	minLevel         int // the minimum cell level to be used.
	maxLevel         int // the maximum cell level to be used.
	levelMod         int // the LevelMod to be used.
	maxCells         int // the maximum desired number of cells in the approximation.
	region           Region
	result           CellUnion
	pq               priorityQueue
	interiorCovering bool // 是否是内部覆盖 cell
}

type candidate struct {
	cell        Cell
	terminal    bool         // Cell should not be expanded further. 是否是终点，如果是终点就不应该继续扩大
	numChildren int          // Number of children that intersect the region. 与该区域相交的children的个数
	children    []*candidate // Actual size may be 0, 4, 16, or 64 elements. 实际大小可能是 0，4，16，64个
	priority    int          // Priority of the candiate. 候选人的优先级
}

// 比较两个数的小的那个
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// 返回两个数的大的那个
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

type priorityQueue []*candidate

func (pq priorityQueue) Len() int {
	return len(pq)
}

func (pq priorityQueue) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].priority > pq[j].priority
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *priorityQueue) Push(x interface{}) {
	item := x.(*candidate)
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() interface{} {
	item := (*pq)[len(*pq)-1]
	*pq = (*pq)[:len(*pq)-1]
	return item
}

func (pq *priorityQueue) Reset() {
	*pq = (*pq)[:0]
}

// newCandidate returns a new candidate with no children if the cell intersects the given region.
// The candidate is marked as terminal if it should not be expanded further.
// newCandidate 函数返回一个新的候选人，这个候选的cell与给定区域相交，那么它就是叶子节点，没有孩子节点。
// 如果它不能进一步的扩展，那么它就要被标记成终点
func (c *coverer) newCandidate(cell Cell) *candidate {
	// 判断当前cell是否和区域相交，不相交就返回nil
	if !c.region.IntersectsCell(cell) {
		return nil
	}
	cand := &candidate{cell: cell}
	level := int(cell.level)
	if level >= c.minLevel {
		if c.interiorCovering {
			if c.region.ContainsCell(cell) {
				cand.terminal = true
			} else if level+c.levelMod > c.maxLevel {
				return nil
			}
		} else if level+c.levelMod > c.maxLevel || c.region.ContainsCell(cell) {
			cand.terminal = true
		}
	}
	return cand
}

// expandChildren populates the children of the candidate by expanding the given number of
// levels from the given cell.  Returns the number of children that were marked "terminal".
func (c *coverer) expandChildren(cand *candidate, cell Cell, numLevels int) int {
	numLevels--
	var numTerminals int
	last := cell.id.ChildEnd()
	for ci := cell.id.ChildBegin(); ci != last; ci = ci.Next() {
		childCell := CellFromCellID(ci)
		if numLevels > 0 {
			if c.region.IntersectsCell(childCell) {
				numTerminals += c.expandChildren(cand, childCell, numLevels)
			}
			continue
		}
		if child := c.newCandidate(childCell); child != nil {
			cand.children = append(cand.children, child)
			cand.numChildren++
			if child.terminal {
				numTerminals++
			}
		}
	}
	return numTerminals
}

// addCandidate adds the given candidate to the result if it is marked as "terminal",
// otherwise expands its children and inserts it into the priority queue.
// Passing an argument of nil does nothing.
func (c *coverer) addCandidate(cand *candidate) {
	if cand == nil {
		return
	}

	if cand.terminal {
		c.result = append(c.result, cand.cell.id)
		return
	}

	// Expand one level at a time until we hit minLevel to ensure that we don't skip over it.
	numLevels := c.levelMod
	level := int(cand.cell.level)
	if level < c.minLevel {
		numLevels = 1
	}

	numTerminals := c.expandChildren(cand, cand.cell, numLevels)
	maxChildrenShift := uint(2 * c.levelMod)
	if cand.numChildren == 0 {
		return
	} else if !c.interiorCovering && numTerminals == 1<<maxChildrenShift && level >= c.minLevel {
		// Optimization: add the parent cell rather than all of its children.
		// We can't do this for interior coverings, since the children just
		// intersect the region, but may not be contained by it - we need to
		// subdivide them further.
		cand.terminal = true
		c.addCandidate(cand)
	} else {
		// We negate the priority so that smaller absolute priorities are returned
		// first. The heuristic is designed to refine the largest cells first,
		// since those are where we have the largest potential gain. Among cells
		// of the same size, we prefer the cells with the fewest children.
		// Finally, among cells with equal numbers of children we prefer those
		// with the smallest number of children that cannot be refined further.
		cand.priority = -(((level<<maxChildrenShift)+cand.numChildren)<<maxChildrenShift + numTerminals)
		heap.Push(&c.pq, cand)
	}
}

// adjustLevel returns the reduced "level" so that it satisfies levelMod. Levels smaller than minLevel
// are not affected (since cells at these levels are eventually expanded).
func (c *coverer) adjustLevel(level int) int {
	if c.levelMod > 1 && level > c.minLevel {
		level -= (level - c.minLevel) % c.levelMod
	}
	return level
}

// adjustCellLevels ensures that all cells with level > minLevel also satisfy levelMod,
// by replacing them with an ancestor if necessary. Cell levels smaller
// than minLevel are not modified (see AdjustLevel). The output is
// then normalized to ensure that no redundant cells are present.
func (c *coverer) adjustCellLevels(cells *CellUnion) {
	if c.levelMod == 1 {
		return
	}

	var out int
	for _, ci := range *cells {
		level := ci.Level()
		newLevel := c.adjustLevel(level)
		if newLevel != level {
			ci = ci.Parent(newLevel)
		}
		if out > 0 && (*cells)[out-1].Contains(ci) {
			continue
		}
		for out > 0 && ci.Contains((*cells)[out-1]) {
			out--
		}
		(*cells)[out] = ci
		out++
	}
	*cells = (*cells)[:out]
}

// initialCandidates computes a set of initial candidates that cover the given region.
func (c *coverer) initialCandidates() {
	// Optimization: start with a small (usually 4 cell) covering of the region's bounding cap.
	temp := &RegionCoverer{MaxLevel: c.maxLevel, LevelMod: 1, MaxCells: min(4, c.maxCells)}

	cells := temp.FastCovering(c.region)
	c.adjustCellLevels(&cells)
	for _, ci := range cells {
		c.addCandidate(c.newCandidate(CellFromCellID(ci)))
	}
}

// coveringInternal generates a covering and stores it in result.
// Strategy: Start with the 6 faces of the cube.  Discard any
// that do not intersect the shape.  Then repeatedly choose the
// largest cell that intersects the shape and subdivide it.
//
// result contains the cells that will be part of the output, while pq
// contains cells that we may still subdivide further. Cells that are
// entirely contained within the region are immediately added to the output,
// while cells that do not intersect the region are immediately discarded.
// Therefore pq only contains cells that partially intersect the region.
// Candidates are prioritized first according to cell size (larger cells
// first), then by the number of intersecting children they have (fewest
// children first), and then by the number of fully contained children
// (fewest children first).

// coveringInternal 方法生成覆盖的结果，并把这个结果存储在 result 中
// 覆盖的策略是：从立方体的6个面开始，不断的舍弃和多边形不相交的 cell。然后不断重复的选择与区域相交的 cell 并不断的将其细分。
//
// result 中的结果仅仅包含最终结果的一部分，优先队列里面包含我们需要继续细分的 cell，也就是边缘附近的 cell。
// 完全包含在区域内的 cell 直接立即添加到结果集中，与这个区域不相交的 cell直接丢弃。

// 因此，优先队列里面只包含部分与区域相交的 cell，也就是边缘附近的 cell。优先队列里面优先级策略，cell 越大优先级越高，
// 即，格子越大，优先级越高，相同大小，拥有相交 cell 孩子数量越少的优先级越高，如果优先级还相同，那么完全包含 cell 孩子数量越少的优先级越高

func (c *coverer) coveringInternal(region Region) {
	c.region = region

	c.initialCandidates()
	for c.pq.Len() > 0 && (!c.interiorCovering || len(c.result) < c.maxCells) {
		cand := heap.Pop(&c.pq).(*candidate)

		// For interior covering we keep subdividing no matter how many children
		// candidate has. If we reach MaxCells before expanding all children,
		// we will just use some of them.
		// For exterior covering we cannot do this, because result has to cover the
		// whole region, so all children have to be used.
		// candidate.numChildren == 1 case takes care of the situation when we
		// already have more then MaxCells in result (minLevel is too high).
		// Subdividing of the candidate with one child does no harm in this case.

		// 不管有多少候选的 cell，都不断的细分它们。如果不断细分孩子，数量达到 MaxCells 的时候，最终我们只使用它们一部分。
		// 对于边缘上的覆盖策略，不能按照上面说的来做，因为结果必须覆盖整个区域，所有的孩子 cell 都要被使用。
		//
		if c.interiorCovering || int(cand.cell.level) < c.minLevel || cand.numChildren == 1 || len(c.result)+c.pq.Len()+cand.numChildren <= c.maxCells {
			for _, child := range cand.children {
				if !c.interiorCovering || len(c.result) < c.maxCells {
					c.addCandidate(child)
				}
			}
		} else {
			cand.terminal = true
			c.addCandidate(cand)
		}
	}
	c.pq.Reset()
	c.region = nil
}

// newCoverer returns an instance of coverer.
func (rc *RegionCoverer) newCoverer() *coverer {
	return &coverer{
		minLevel: max(0, min(maxLevel, rc.MinLevel)),
		maxLevel: max(0, min(maxLevel, rc.MaxLevel)),
		levelMod: max(1, min(3, rc.LevelMod)),
		maxCells: rc.MaxCells,
	}
}

// Covering returns a CellUnion that covers the given region and satisfies the various restrictions.
func (rc *RegionCoverer) Covering(region Region) CellUnion {
	covering := rc.CellUnion(region)
	covering.Denormalize(max(0, min(maxLevel, rc.MinLevel)), max(1, min(3, rc.LevelMod)))
	return covering
}

// InteriorCovering returns a CellUnion that is contained within the given region and satisfies the various restrictions.
func (rc *RegionCoverer) InteriorCovering(region Region) CellUnion {
	intCovering := rc.InteriorCellUnion(region)
	intCovering.Denormalize(max(0, min(maxLevel, rc.MinLevel)), max(1, min(3, rc.LevelMod)))
	return intCovering
}

// CellUnion returns a normalized CellUnion that covers the given region and
// satisfies the restrictions except for minLevel and levelMod. These criteria
// cannot be satisfied using a cell union because cell unions are
// automatically normalized by replacing four child cells with their parent
// whenever possible. (Note that the list of cell ids passed to the CellUnion
// constructor does in fact satisfy all the given restrictions.)
func (rc *RegionCoverer) CellUnion(region Region) CellUnion {
	c := rc.newCoverer()
	c.coveringInternal(region)
	cu := c.result
	cu.Normalize()
	return cu
}

// InteriorCellUnion returns a normalized CellUnion that is contained within the given region and
// satisfies the restrictions except for minLevel and levelMod. These criteria
// cannot be satisfied using a cell union because cell unions are
// automatically normalized by replacing four child cells with their parent
// whenever possible. (Note that the list of cell ids passed to the CellUnion
// constructor does in fact satisfy all the given restrictions.)
func (rc *RegionCoverer) InteriorCellUnion(region Region) CellUnion {
	c := rc.newCoverer()
	c.interiorCovering = true
	c.coveringInternal(region)
	cu := c.result
	cu.Normalize()
	return cu
}

// FastCovering returns a CellUnion that covers the given region similar to Covering,
// except that this method is much faster and the coverings are not as tight.
// All of the usual parameters are respected (MaxCells, MinLevel, MaxLevel, and LevelMod),
// except that the implementation makes no attempt to take advantage of large values of
// MaxCells.  (A small number of cells will always be returned.)
//
// This function is useful as a starting point for algorithms that
// recursively subdivide cells.

// FastCovering 函数返回一个 CellUnion 集合，这个集合里面的 cell 覆盖了给定的区域，不同之处在于，这个方法速度很快，
// 得到的结果也比较粗糙。当然得到的 CellUnion 集合也是满足 MaxCells, MinLevel, MaxLevel, 和 LevelMod 要求的。
// 只不过结果不尝试去使用 MaxCells 的大值。一般会返回少量的 cell，所以结果比较粗糙。

// 这个函数作为递归细分 cell 的起点，非常管用。
func (rc *RegionCoverer) FastCovering(region Region) CellUnion {
	c := rc.newCoverer()
	cu := CellUnion(region.CellUnionBound())
	c.normalizeCovering(&cu)
	return cu
}

// normalizeCovering normalizes the "covering" so that it conforms to the current covering
// parameters (MaxCells, minLevel, maxLevel, and levelMod).
// This method makes no attempt to be optimal. In particular, if
// minLevel > 0 or levelMod > 1 then it may return more than the
// desired number of cells even when this isn't necessary.
//
// Note that when the covering parameters have their default values, almost
// all of the code in this function is skipped.
func (c *coverer) normalizeCovering(covering *CellUnion) {
	// If any cells are too small, or don't satisfy levelMod, then replace them with ancestors.
	if c.maxLevel < maxLevel || c.levelMod > 1 {
		for i, ci := range *covering {
			level := ci.Level()
			newLevel := c.adjustLevel(min(level, c.maxLevel))
			if newLevel != level {
				(*covering)[i] = ci.Parent(newLevel)
			}
		}
	}
	// Sort the cells and simplify them.
	covering.Normalize()

	// If there are still too many cells, then repeatedly replace two adjacent
	// cells in CellID order by their lowest common ancestor.
	for len(*covering) > c.maxCells {
		bestIndex := -1
		bestLevel := -1
		for i := 0; i+1 < len(*covering); i++ {
			level, ok := (*covering)[i].CommonAncestorLevel((*covering)[i+1])
			if !ok {
				continue
			}
			level = c.adjustLevel(level)
			if level > bestLevel {
				bestLevel = level
				bestIndex = i
			}
		}

		if bestLevel < c.minLevel {
			break
		}
		(*covering)[bestIndex] = (*covering)[bestIndex].Parent(bestLevel)
		covering.Normalize()
	}
	// Make sure that the covering satisfies minLevel and levelMod,
	// possibly at the expense of satisfying MaxCells.
	if c.minLevel > 0 || c.levelMod > 1 {
		covering.Denormalize(c.minLevel, c.levelMod)
	}
}

// BUG(akashagrawal): The differences from the C++ version FloodFill, SimpleCovering
