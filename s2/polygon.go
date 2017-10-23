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
	"fmt"
	"io"
)

// Polygon represents a sequence of zero or more loops; recall that the
// interior of a loop is defined to be its left-hand side (see Loop).
//
// When the polygon is initialized, the given loops are automatically converted
// into a canonical form consisting of "shells" and "holes". Shells and holes
// are both oriented CCW, and are nested hierarchically. The loops are
// reordered to correspond to a pre-order traversal of the nesting hierarchy.
//
// Polygons may represent any region of the sphere with a polygonal boundary,
// including the entire sphere (known as the "full" polygon). The full polygon
// consists of a single full loop (see Loop), whereas the empty polygon has no
// loops at all.
//
// Use FullPolygon() to construct a full polygon. The zero value of Polygon is
// treated as the empty polygon.
//
// Polygons have the following restrictions:
//
//  - Loops may not cross, i.e. the boundary of a loop may not intersect
//    both the interior and exterior of any other loop.
//
//  - Loops may not share edges, i.e. if a loop contains an edge AB, then
//    no other loop may contain AB or BA.
//
//  - Loops may share vertices, however no vertex may appear twice in a
//    single loop (see Loop).
//
//  - No loop may be empty. The full loop may appear only in the full polygon.
type Polygon struct {
	loops []*Loop

	// index is a spatial index of all the polygon loops.
	index *ShapeIndex

	// hasHoles tracks if this polygon has at least one hole.
	hasHoles bool

	// numVertices keeps the running total of all of the vertices of the contained loops.
	numVertices int

	// numEdges tracks the total number of edges in all the loops in this polygon.
	numEdges int

	// bound is a conservative bound on all points contained by this loop.
	// If l.ContainsPoint(P), then l.bound.ContainsPoint(P).
	bound Rect

	// Since bound is not exact, it is possible that a loop A contains
	// another loop B whose bounds are slightly larger. subregionBound
	// has been expanded sufficiently to account for this error, i.e.
	// if A.Contains(B), then A.subregionBound.Contains(B.bound).
	subregionBound Rect

	// A slice where element i is the cumulative number of edges in the
	// preceding loops in the polygon. This field is used for polygons that
	// have a large number of loops, and may be empty for polygons with few loops.
	cumulativeEdges []int
}

// PolygonFromLoops constructs a polygon from the given set of loops. The polygon
// interior consists of the points contained by an odd number of loops. (Recall
// that a loop contains the set of points on its left-hand side.)
//
// This method determines the loop nesting hierarchy and assigns every loop a
// depth. Shells have even depths, and holes have odd depths.
//
// Note: The given set of loops are reordered by this method so that the hierarchy
// can be traversed using Parent, LastDescendant and the loops depths.
func PolygonFromLoops(loops []*Loop) *Polygon {
	p := &Polygon{}
	// Empty polygons do not contain any loops, even the Empty loop.
	if len(loops) == 1 && loops[0].IsEmpty() {
		p.initLoopProperties()
		return p
	}
	p.loops = loops
	p.initNested()
	return p

}

// TODO(roberts): Implement initOriented
/*
// PolygonFromOrientedLoops, like PolygonFromLoops, returns a Polygon from the
// given set of loops. It expects loops to be oriented such that the polygon
// interior is on the left-hand side of all loops. This implies that shells
// and holes should have opposite orientations in the input to this method.
// (During initialization, loops representing holes will automatically be
// inverted.)
func PolygonFromOrientedLoops(loops []*Loop) *Polygon {
	panic("PolygonFromOrientedLoops not yet implemented")
	p := &Polygon{
		loops: loops,
	}
	p.initOriented()
	return p
}
*/

// PolygonFromCell returns a Polygon from a single loop created from the given Cell.
func PolygonFromCell(cell Cell) *Polygon {
	return PolygonFromLoops([]*Loop{LoopFromCell(cell)})
}

// initNested takes the set of loops in this polygon and performs the nesting
// computations to set the proper nesting and parent/child relationships.
func (p *Polygon) initNested() {
	if len(p.loops) == 1 {
		p.initOneLoop()
		return
	}

	lm := make(loopMap)

	for _, l := range p.loops {
		lm.insertLoop(l, nil)
	}

	// Reorder the loops in depth-first traversal order.
	p.initLoops(lm)
	p.initLoopProperties()
}

// initOriented takes the loops in this polygon and performs the nesting
// computations. It expects the loops to be oriented such that the polygon
// interior is on the left-hand side of all loops. This implies that shells
// and holes should have opposite orientations in the input to this method.
// (During initialization, loops representing holes will automatically be
// inverted.)
func (p *Polygon) initOriented() {
	// Here is the algorithm:
	//
	// 1. Remember which of the given loops contain OriginPoint.
	//
	// 2. Invert loops as necessary to ensure that they are nestable (i.e., no
	//    loop contains the complement of any other loop). This may result in a
	//    set of loops corresponding to the complement of the given polygon, but
	//    we will fix that problem later.
	//
	//    We make the loops nestable by first normalizing all the loops (i.e.,
	//    inverting any loops whose turning angle is negative). This handles
	//    all loops except those whose turning angle is very close to zero
	//    (within the maximum error tolerance). Any such loops are inverted if
	//    and only if they contain OriginPoint(). (In theory this step is only
	//    necessary if there are at least two such loops.) The resulting set of
	//    loops is guaranteed to be nestable.
	//
	// 3. Build the polygon. This yields either the desired polygon or its
	//    complement.
	//
	// 4. If there is at least one loop, we find a loop L that is adjacent to
	//    OriginPoint() (where "adjacent" means that there exists a path
	//    connecting OriginPoint() to some vertex of L such that the path does
	//    not cross any loop). There may be a single such adjacent loop, or
	//    there may be several (in which case they should all have the same
	//    contains_origin() value). We choose L to be the loop containing the
	//    origin whose depth is greatest, or loop(0) (a top-level shell) if no
	//    such loop exists.
	//
	// 5. If (L originally contained origin) != (polygon contains origin), we
	//    invert the polygon. This is done by inverting a top-level shell whose
	//    turning angle is minimal and then fixing the nesting hierarchy. Note
	//    that because we normalized all the loops initially, this step is only
	//    necessary if the polygon requires at least one non-normalized loop to
	//    represent it.
	panic("initOriented not yet implemented")
}

// loopMap is a map of a loop to its immediate children with respect to nesting.
// It is used to determine which loops are shells and which are holes.
type loopMap map[*Loop][]*Loop

// insertLoop adds the given loop to the loop map under the specified parent.
// All children of the new entry are checked to see if the need to move up to
// a different level.
func (lm loopMap) insertLoop(newLoop, parent *Loop) {
	var children []*Loop
	for done := false; !done; {
		children = lm[parent]
		done = true
		for _, child := range children {
			if child.ContainsNested(newLoop) {
				parent = child
				done = false
				break
			}
		}
	}

	// Now, we have found a parent for this loop, it may be that some of the
	// children of the parent of this loop may now be children of the new loop.
	newChildren := lm[newLoop]
	for i := 0; i < len(children); {
		child := children[i]
		if newLoop.ContainsNested(child) {
			newChildren = append(newChildren, child)
			children = append(children[0:i], children[i+1:]...)
		} else {
			i++
		}
	}

	lm[newLoop] = newChildren
	lm[parent] = append(children, newLoop)
}

// loopStack simplifies access to the loops while being initialized.
type loopStack []*Loop

func (s *loopStack) push(v *Loop) {
	*s = append(*s, v)
}
func (s *loopStack) pop() *Loop {
	l := len(*s)
	r := (*s)[l-1]
	*s = (*s)[:l-1]
	return r
}

// initLoops walks the mapping of loops to all of their children, and adds them in
// order into to the polygons set of loops.
func (p *Polygon) initLoops(lm loopMap) {
	var stack loopStack
	stack.push(nil)
	depth := -1

	for len(stack) > 0 {
		loop := stack.pop()
		if loop != nil {
			depth = loop.depth
			p.loops = append(p.loops, loop)
		}
		children := lm[loop]
		for i := len(children) - 1; i >= 0; i-- {
			child := children[i]
			child.depth = depth + 1
			stack.push(child)
		}
	}
}

// initOneLoop set the properties for a polygon made of a single loop.
// TODO(roberts): Can this be merged with initLoopProperties
func (p *Polygon) initOneLoop() {
	p.hasHoles = false
	p.numVertices = len(p.loops[0].vertices)
	p.bound = p.loops[0].RectBound()
	p.subregionBound = ExpandForSubregions(p.bound)
	// Ensure the loops depth is set correctly.
	p.loops[0].depth = 0

	p.initEdgesAndIndex()
}

// initLoopProperties sets the properties for polygons with multiple loops.
func (p *Polygon) initLoopProperties() {
	// the loops depths are set by initNested/initOriented prior to this.

	p.hasHoles = false
	for k, l := range p.loops {
		if p.loopIsHole(k) {
			p.hasHoles = true
		} else {
			p.bound = p.bound.Union(l.RectBound())
		}
		p.numVertices += l.NumVertices()
	}
	p.subregionBound = ExpandForSubregions(p.bound)

	p.initEdgesAndIndex()
}

// initEdgesAndIndex performs the shape related initializations and adds the final
// polygon to the index.
func (p *Polygon) initEdgesAndIndex() {
	if p.IsFull() {
		return
	}
	const maxLinearSearchLoops = 12 // Based on benchmarks.
	if len(p.loops) > maxLinearSearchLoops {
		p.cumulativeEdges = make([]int, 0, len(p.loops))
	}

	for _, l := range p.loops {
		if p.cumulativeEdges != nil {
			p.cumulativeEdges = append(p.cumulativeEdges, p.numEdges)
		}
		p.numEdges += len(l.vertices)
	}

	p.index = NewShapeIndex()
	p.index.Add(p)
}

// FullPolygon returns a special "full" polygon.
func FullPolygon() *Polygon {
	ret := &Polygon{
		loops: []*Loop{
			FullLoop(),
		},
		numVertices:    len(FullLoop().Vertices()),
		bound:          FullRect(),
		subregionBound: FullRect(),
	}
	ret.initEdgesAndIndex()
	return ret
}

// IsEmpty reports whether this is the special "empty" polygon (consisting of no loops).
func (p *Polygon) IsEmpty() bool {
	return len(p.loops) == 0
}

// IsFull reports whether this is the special "full" polygon (consisting of a
// single loop that encompasses the entire sphere).
func (p *Polygon) IsFull() bool {
	return len(p.loops) == 1 && p.loops[0].IsFull()
}

// NumLoops returns the number of loops in this polygon.
func (p *Polygon) NumLoops() int {
	return len(p.loops)
}

// Loops returns the loops in this polygon.
func (p *Polygon) Loops() []*Loop {
	return p.loops
}

// Loop returns the loop at the given index. Note that during initialization,
// the given loops are reordered according to a pre-order traversal of the loop
// nesting hierarchy. This implies that every loop is immediately followed by
// its descendants. This hierarchy can be traversed using the methods Parent,
// LastDescendant, and Loop.depth.
func (p *Polygon) Loop(k int) *Loop {
	return p.loops[k]
}

// Parent returns the index of the parent of loop k.
// If the loop does not have a parent, ok=false is returned.
func (p *Polygon) Parent(k int) (index int, ok bool) {
	// See where we are on the depth hierarchy.
	depth := p.loops[k].depth
	if depth == 0 {
		return -1, false
	}

	// There may be several loops at the same nesting level as us that share a
	// parent loop with us. (Imagine a slice of swiss cheese, of which we are one loop.
	// we don't know how many may be next to us before we get back to our parent loop.)
	// Move up one position from us, and then begin traversing back through the set of loops
	// until we find the one that is our parent or we get to the top of the polygon.
	for k--; k >= 0 && p.loops[k].depth <= depth; k-- {
	}
	return k, true
}

// LastDescendant returns the index of the last loop that is contained within loop k.
// If k is negative, it returns the last loop in the polygon.
// Note that loops are indexed according to a pre-order traversal of the nesting
// hierarchy, so the immediate children of loop k can be found by iterating over
// the loops (k+1)..LastDescendant(k) and selecting those whose depth is equal
// to Loop(k).depth+1.
func (p *Polygon) LastDescendant(k int) int {
	if k < 0 {
		return len(p.loops) - 1
	}

	depth := p.loops[k].depth

	// Find the next loop immediately past us in the set of loops, and then start
	// moving down the list until we either get to the end or find the next loop
	// that is higher up the hierarchy than we are.
	for k++; k < len(p.loops) && p.loops[k].depth > depth; k++ {
	}
	return k - 1
}

// loopIsHole reports whether the given loop represents a hole in this polygon.
func (p *Polygon) loopIsHole(k int) bool {
	return p.loops[k].depth&1 != 0
}

// loopSign returns -1 if this loop represents a hole in this polygon.
// Otherwise, it returns +1. This is used when computing the area of a polygon.
// (holes are subtracted from the total area).
func (p *Polygon) loopSign(k int) int {
	if p.loopIsHole(k) {
		return -1
	}
	return 1
}

// CapBound returns a bounding spherical cap.
func (p *Polygon) CapBound() Cap { return p.bound.CapBound() }

// RectBound returns a bounding latitude-longitude rectangle.
func (p *Polygon) RectBound() Rect { return p.bound }

// ContainsPoint reports whether the polygon contains the point.
func (p *Polygon) ContainsPoint(point Point) bool {
	// NOTE: A bounds check slows down this function by about 50%. It is
	// worthwhile only when it might allow us to delay building the index.
	if !p.index.IsFresh() && !p.bound.ContainsPoint(point) {
		return false
	}

	// For small polygons, and during initial construction, it is faster to just
	// check all the crossing.
	const maxBruteForceVertices = 32
	if p.numVertices < maxBruteForceVertices || p.index == nil {
		inside := false
		for _, l := range p.loops {
			// use loops bruteforce to avoid building the index on each loop.
			inside = inside != l.bruteForceContainsPoint(point)
		}
		return inside
	}

	// Otherwise, look up the ShapeIndex cell containing this point.
	it := p.index.Iterator()
	if !it.LocatePoint(point) {
		return false
	}

	return p.iteratorContainsPoint(it, point)
}

// ContainsCell reports whether the polygon contains the given cell.
func (p *Polygon) ContainsCell(cell Cell) bool {
	it := p.index.Iterator()
	relation := it.LocateCellID(cell.ID())

	// If "cell" is disjoint from all index cells, it is not contained.
	// Similarly, if "cell" is subdivided into one or more index cells then it
	// is not contained, since index cells are subdivided only if they (nearly)
	// intersect a sufficient number of edges.  (But note that if "cell" itself
	// is an index cell then it may be contained, since it could be a cell with
	// no edges in the loop interior.)
	if relation != Indexed {
		return false
	}

	// Otherwise check if any edges intersect "cell".
	if p.boundaryApproxIntersects(it, cell) {
		return false
	}

	// Otherwise check if the loop contains the center of "cell".
	return p.iteratorContainsPoint(it, cell.Center())
}

// IntersectsCell reports whether the polygon intersects the given cell.
func (p *Polygon) IntersectsCell(cell Cell) bool {
	it := p.index.Iterator()
	relation := it.LocateCellID(cell.ID())

	// If cell does not overlap any index cell, there is no intersection.
	if relation == Disjoint {
		return false
	}
	// If cell is subdivided into one or more index cells, there is an
	// intersection to within the S2ShapeIndex error bound (see Contains).
	if relation == Subdivided {
		return true
	}
	// If cell is an index cell, there is an intersection because index cells
	// are created only if they have at least one edge or they are entirely
	// contained by the loop.
	if it.CellID() == cell.id {
		return true
	}
	// Otherwise check if any edges intersect cell.
	if p.boundaryApproxIntersects(it, cell) {
		return true
	}
	// Otherwise check if the loop contains the center of cell.
	return p.iteratorContainsPoint(it, cell.Center())
}

// CellUnionBound computes a covering of the Polygon.
func (p *Polygon) CellUnionBound() []CellID {
	// TODO(roberts): Use ShapeIndexRegion when it's available.
	return p.CapBound().CellUnionBound()
}

// boundaryApproxIntersects reports whether the loop's boundary intersects cell.
// It may also return true when the loop boundary does not intersect cell but
// some edge comes within the worst-case error tolerance.
//
// This requires that it.Locate(cell) returned Indexed.
func (p *Polygon) boundaryApproxIntersects(it *ShapeIndexIterator, cell Cell) bool {
	aClipped := it.IndexCell().findByShapeID(0)

	// If there are no edges, there is no intersection.
	if len(aClipped.edges) == 0 {
		return false
	}

	// We can save some work if cell is the index cell itself.
	if it.CellID() == cell.ID() {
		return true
	}

	// Otherwise check whether any of the edges intersect cell.
	maxError := (faceClipErrorUVCoord + intersectsRectErrorUVDist)
	bound := cell.BoundUV().ExpandedByMargin(maxError)
	for _, e := range aClipped.edges {
		edge := p.index.Shape(0).Edge(e)
		v0, v1, ok := ClipToPaddedFace(edge.V0, edge.V1, cell.Face(), maxError)
		if ok && edgeIntersectsRect(v0, v1, bound) {
			return true
		}
	}

	return false
}

// iteratorContainsPoint reports whether the iterator that is positioned at the
// ShapeIndexCell that may contain p, contains the point p.
func (p *Polygon) iteratorContainsPoint(it *ShapeIndexIterator, point Point) bool {
	// Test containment by drawing a line segment from the cell center to the
	// given point and counting edge crossings.
	aClipped := it.IndexCell().findByShapeID(0)
	inside := aClipped.containsCenter

	if len(aClipped.edges) == 0 {
		return inside
	}

	// This block requires ShapeIndex.
	crosser := NewEdgeCrosser(it.Center(), point)
	shape := p.index.Shape(0)
	for _, e := range aClipped.edges {
		edge := shape.Edge(e)
		inside = inside != crosser.EdgeOrVertexCrossing(edge.V0, edge.V1)
	}

	return inside
}

// Shape Interface

// NumEdges returns the number of edges in this shape.
func (p *Polygon) NumEdges() int {
	return p.numEdges
}

// Edge returns endpoints for the given edge index.
func (p *Polygon) Edge(e int) Edge {
	var i int

	if len(p.cumulativeEdges) > 0 {
		for i = range p.cumulativeEdges {
			if i+1 >= len(p.cumulativeEdges) || e < p.cumulativeEdges[i+1] {
				e -= p.cumulativeEdges[i]
				break
			}
		}
	} else {
		// When the number of loops is small, use linear search. Most often
		// there is exactly one loop and the code below executes zero times.
		for i = 0; e >= len(p.Loop(i).vertices); i++ {
			e -= len(p.Loop(i).vertices)
		}
	}

	return Edge{p.Loop(i).OrientedVertex(e), p.Loop(i).OrientedVertex(e + 1)}
}

// HasInterior reports whether this Polygon has an interior.
func (p *Polygon) HasInterior() bool {
	return p.dimension() == polygonGeometry
}

// ReferencePoint returns the reference point for this polygon.
func (p *Polygon) ReferencePoint() ReferencePoint {
	containsOrigin := false
	for _, l := range p.loops {
		containsOrigin = containsOrigin != l.ContainsOrigin()
	}
	return OriginReferencePoint(containsOrigin)
}

// NumChains reports the number of contiguous edge chains in the Polygon.
func (p *Polygon) NumChains() int {
	if p.IsFull() {
		return 0
	}

	return p.NumLoops()
}

// Chain returns the i-th edge Chain (loop) in the Shape.
func (p *Polygon) Chain(chainID int) Chain {
	if p.cumulativeEdges != nil {
		return Chain{p.cumulativeEdges[chainID], len(p.Loop(chainID).vertices)}
	}
	e := 0
	for j := 0; j < chainID; j++ {
		e += len(p.Loop(j).vertices)
	}
	return Chain{e, len(p.Loop(chainID).vertices)}
}

// ChainEdge returns the j-th edge of the i-th edge Chain (loop).
func (p *Polygon) ChainEdge(i, j int) Edge {
	return Edge{p.Loop(i).OrientedVertex(j), p.Loop(i).OrientedVertex(j + 1)}
}

// ChainPosition returns a pair (i, j) such that edgeID is the j-th edge
// of the i-th edge Chain.
func (p *Polygon) ChainPosition(edgeID int) ChainPosition {
	var i int

	if len(p.cumulativeEdges) > 0 {
		for i = range p.cumulativeEdges {
			if i+1 >= len(p.cumulativeEdges) || edgeID < p.cumulativeEdges[i+1] {
				edgeID -= p.cumulativeEdges[i]
				break
			}
		}
	} else {
		// When the number of loops is small, use linear search. Most often
		// there is exactly one loop and the code below executes zero times.
		for i = 0; edgeID >= len(p.Loop(i).vertices); i++ {
			edgeID -= len(p.Loop(i).vertices)
		}
	}
	// TODO(roberts): unify this and Edge since they are mostly identical.
	return ChainPosition{i, edgeID}
}

// dimension returns the dimension of the geometry represented by this Polygon.
func (p *Polygon) dimension() dimension { return polygonGeometry }

// Encode encodes the Polygon
func (p *Polygon) Encode(w io.Writer) error {
	e := &encoder{w: w}
	p.encode(e)
	return e.err
}

// encode only supports lossless encoding and not compressed format.
func (p *Polygon) encode(e *encoder) {
	if p.numVertices == 0 {
		p.encodeCompressed(e, maxLevel, nil)
		return
	}

	// Convert all the polygon vertices to XYZFaceSiTi format.
	vs := make([]xyzFaceSiTi, 0, p.numVertices)
	for _, l := range p.loops {
		vs = append(vs, l.xyzFaceSiTiVertices()...)
	}

	// Computes a histogram of the cell levels at which the vertices are snapped.
	// (histogram[0] is the number of unsnapped vertices, histogram[i] the number
	// of vertices snapped at level i-1).
	histogram := make([]int, maxLevel+2)
	for _, v := range vs {
		histogram[v.level+1]++
	}

	// Compute the level at which most of the vertices are snapped.
	// If multiple levels have the same maximum number of vertices
	// snapped to it, the first one (lowest level number / largest
	// area / smallest encoding length) will be chosen, so this
	// is desired.
	var snapLevel, numSnapped int
	for level, h := range histogram[1:] {
		if h > numSnapped {
			snapLevel, numSnapped = level, h
		}
	}

	// Choose an encoding format based on the number of unsnapped vertices and a
	// rough estimate of the encoded sizes.
	numUnsnapped := p.numVertices - numSnapped // Number of vertices that won't be snapped at snapLevel.
	const pointSize = 3 * 8                    // s2.Point is an r3.Vector, which is 3 float64s. That's 3*8 = 24 bytes.
	compressedSize := 4*p.numVertices + (pointSize+2)*numUnsnapped
	losslessSize := pointSize * p.numVertices
	if compressedSize < losslessSize {
		p.encodeCompressed(e, snapLevel, vs)
	} else {
		p.encodeLossless(e)
	}
}

// encodeLossless encodes the polygon's Points as float64s.
func (p *Polygon) encodeLossless(e *encoder) {
	e.writeInt8(encodingVersion)
	e.writeBool(true) // a legacy c++ value. must be true.
	e.writeBool(p.hasHoles)
	e.writeUint32(uint32(len(p.loops)))

	if e.err != nil {
		return
	}
	if len(p.loops) > maxEncodedLoops {
		e.err = fmt.Errorf("too many loops (%d; max is %d)", len(p.loops), maxEncodedLoops)
		return
	}
	for _, l := range p.loops {
		l.encode(e)
	}

	// Encode the bound.
	p.bound.encode(e)
}

func (p *Polygon) encodeCompressed(e *encoder, snapLevel int, vertices []xyzFaceSiTi) {
	e.writeUint8(uint8(encodingCompressedVersion))
	e.writeUint8(uint8(snapLevel))
	e.writeUvarint(uint64(len(p.loops)))

	if e.err != nil {
		return
	}
	if l := len(p.loops); l > maxEncodedLoops {
		e.err = fmt.Errorf("too many loops to encode: %d; max is %d", l, maxEncodedLoops)
		return
	}

	for _, l := range p.loops {
		l.encodeCompressed(e, snapLevel, vertices[:len(l.vertices)])
		vertices = vertices[len(l.vertices):]
	}
	// Do not write the bound, num_vertices, or has_holes_ as they can be
	// cheaply recomputed by decodeCompressed.  Microbenchmarks show the
	// speed difference is inconsequential.
}

// Decode decodes the Polygon.
func (p *Polygon) Decode(r io.Reader) error {
	d := &decoder{r: asByteReader(r)}
	version := int8(d.readUint8())
	var dec func(*decoder)
	switch version {
	case encodingVersion:
		dec = p.decode
	case encodingCompressedVersion:
		dec = p.decodeCompressed
	default:
		return fmt.Errorf("unsupported version %d", version)
	}
	dec(d)
	return d.err
}

// maxEncodedLoops is the biggest supported number of loops in a polygon during encoding.
// Setting a maximum guards an allocation: it prevents an attacker from easily pushing us OOM.
const maxEncodedLoops = 10000000

func (p *Polygon) decode(d *decoder) {
	*p = Polygon{}
	d.readUint8() // Ignore irrelevant serialized owns_loops_ value.

	p.hasHoles = d.readBool()

	// Polygons with no loops are explicitly allowed here: a newly created
	// polygon has zero loops and such polygons encode and decode properly.
	nloops := d.readUint32()
	if d.err != nil {
		return
	}
	if nloops > maxEncodedLoops {
		d.err = fmt.Errorf("too many loops (%d; max is %d)", nloops, maxEncodedLoops)
		return
	}
	p.loops = make([]*Loop, nloops)
	for i := range p.loops {
		p.loops[i] = new(Loop)
		p.loops[i].decode(d)
		p.numVertices += len(p.loops[i].vertices)
	}

	p.bound.decode(d)
	if d.err != nil {
		return
	}
	p.subregionBound = ExpandForSubregions(p.bound)
	p.initEdgesAndIndex()
}

func (p *Polygon) decodeCompressed(d *decoder) {
	snapLevel := int(d.readUint8())

	if snapLevel > maxLevel {
		d.err = fmt.Errorf("snaplevel too big: %d", snapLevel)
		return
	}
	// Polygons with no loops are explicitly allowed here: a newly created
	// polygon has zero loops and such polygons encode and decode properly.
	nloops := int(d.readUvarint())
	if nloops > maxEncodedLoops {
		d.err = fmt.Errorf("too many loops (%d; max is %d)", nloops, maxEncodedLoops)
	}
	p.loops = make([]*Loop, nloops)
	for i := range p.loops {
		p.loops[i] = new(Loop)
		p.loops[i].decodeCompressed(d, snapLevel)
		// TODO(roberts): Update this bound.Union call when initLoopProperties is implemented.
		p.bound = p.bound.Union(p.loops[i].bound)
		p.numVertices += len(p.loops[i].vertices)
	}
	if d.err != nil {
		return
	}
	if p.numVertices == 0 {
		p.bound = EmptyRect()
	}
	p.subregionBound = ExpandForSubregions(p.bound)
	p.initEdgesAndIndex()
}

// TODO(roberts): Differences from C++
// IsValid
// Area
// Centroid
// SnapLevel
// DistanceToPoint
// DistanceToBoundary
// Project
// ProjectToBoundary
// Contains/ApproxContains/Intersects/ApproxDisjoint for Polygons
// InitTo{Intersection/ApproxIntersection/Union/ApproxUnion/Diff/ApproxDiff}
// InitToSimplified
// InitToSnapped
// IntersectWithPolyline
// ApproxIntersectWithPolyline
// SubtractFromPolyline
// ApproxSubtractFromPolyline
// DestructiveUnion
// DestructiveApproxUnion
// InitToCellUnionBorder
// IsNormalized
// Equals/BoundaryEquals/BoundaryApproxEquals/BoundaryNear Polygons
// BreakEdgesAndAddToBuilder
//
// clearLoops
// findLoopNestingError
// initToSimplifiedInternal
// internalClipPolyline
// compareBoundary
// containsBoundary
// excludesBoundary
// containsNonCrossingBoundary
// excludesNonCrossingShells
// anyLoopContains(Loop)
// anyLoopIntersects(Loop)
// clipBoundary
