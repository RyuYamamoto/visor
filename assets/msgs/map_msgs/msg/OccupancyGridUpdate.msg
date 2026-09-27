# Update to an existing OccupancyGrid: a rectangular patch of cells.
std_msgs/Header header
int32 x
int32 y
uint32 width
uint32 height

# The map patch data, row-major, for the region [x, x+width) x [y, y+height).
int8[] data
