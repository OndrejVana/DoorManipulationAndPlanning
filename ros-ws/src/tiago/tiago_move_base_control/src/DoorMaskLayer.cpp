#include <tiago_move_base_control/DoorMaskLayer.h>

#include <geometry_msgs/PointStamped.h>
#include <pluginlib/class_list_macros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <algorithm>
#include <limits>

PLUGINLIB_EXPORT_CLASS(tiago_move_base_control::DoorMaskLayer, costmap_2d::Layer)

namespace tiago_move_base_control
{

DoorMaskLayer::DoorMaskLayer()
: set_unknown_(false),
  have_polygon_(false),
  enabled_(false),
  repaint_(false),
  repaint_min_x_(0), repaint_min_y_(0), repaint_max_x_(0), repaint_max_y_(0)
{}

void DoorMaskLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);

  nh.param("polygon_topic", polygon_topic_, std::string("/door_mask/polygon"));
  nh.param("enabled_topic", enabled_topic_, std::string("/door_mask/enabled"));
  nh.param("set_unknown_instead_of_free", set_unknown_, false);

  polygon_sub_ = nh.subscribe(polygon_topic_, 1, &DoorMaskLayer::polygonCb, this);
  enabled_sub_ = nh.subscribe(enabled_topic_, 1, &DoorMaskLayer::enabledCb, this);

  current_ = true;

  ROS_INFO_STREAM("DoorMaskLayer: polygon " << polygon_topic_ << ", enabled by " << enabled_topic_);
}

void DoorMaskLayer::enabledCb(const std_msgs::BoolConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (enabled_ && !msg->data)
  {
    repaint_ = !masked_.empty();
  }
  enabled_ = msg->data;
}

void DoorMaskLayer::polygonCb(const geometry_msgs::PolygonStampedConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  repaint_ = repaint_ || !masked_.empty();
  polygon_ = *msg;
  have_polygon_ = msg->polygon.points.size() >= 3;
}

bool DoorMaskLayer::polygonInCostmapFrame(std::vector<Point>& points)
{
  const std::string frame = layered_costmap_->getGlobalFrameID();
  geometry_msgs::TransformStamped transform;
  try
  {
    transform = tf_->lookupTransform(frame, polygon_.header.frame_id, ros::Time(0));
  }
  catch (const tf2::TransformException& error)
  {
    ROS_WARN_THROTTLE(5.0, "DoorMaskLayer: no transform %s -> %s: %s",
                      polygon_.header.frame_id.c_str(), frame.c_str(), error.what());
    return false;
  }
  points.clear();
  for (const auto& corner : polygon_.polygon.points)
  {
    geometry_msgs::Point in, out;
    in.x = corner.x;
    in.y = corner.y;
    tf2::doTransform(in, out, transform);
    points.push_back({out.x, out.y});
  }
  return true;
}

void DoorMaskLayer::updateBounds(double, double, double,
                                 double* min_x, double* min_y, double* max_x, double* max_y)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (repaint_)
  {
    *min_x = std::min(*min_x, repaint_min_x_);
    *min_y = std::min(*min_y, repaint_min_y_);
    *max_x = std::max(*max_x, repaint_max_x_);
    *max_y = std::max(*max_y, repaint_max_y_);
    repaint_ = false;
    masked_.clear();
  }
  if (!enabled_ || !have_polygon_ || !polygonInCostmapFrame(masked_))
  {
    masked_.clear();
    return;
  }
  repaint_min_x_ = repaint_min_y_ = std::numeric_limits<double>::max();
  repaint_max_x_ = repaint_max_y_ = std::numeric_limits<double>::lowest();
  for (const auto& point : masked_)
  {
    repaint_min_x_ = std::min(repaint_min_x_, point.x);
    repaint_min_y_ = std::min(repaint_min_y_, point.y);
    repaint_max_x_ = std::max(repaint_max_x_, point.x);
    repaint_max_y_ = std::max(repaint_max_y_, point.y);
  }
  *min_x = std::min(*min_x, repaint_min_x_);
  *min_y = std::min(*min_y, repaint_min_y_);
  *max_x = std::max(*max_x, repaint_max_x_);
  *max_y = std::max(*max_y, repaint_max_y_);
}

bool DoorMaskLayer::inside(const std::vector<Point>& polygon, double x, double y)
{
  bool result = false;
  for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
  {
    const Point& a = polygon[i];
    const Point& b = polygon[j];
    if ((a.y > y) != (b.y > y) && x < a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y))
    {
      result = !result;
    }
  }
  return result;
}

void DoorMaskLayer::updateCosts(costmap_2d::Costmap2D& master_grid,
                                int min_i, int min_j, int max_i, int max_j)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (masked_.empty())
  {
    return;
  }
  min_i = std::max(min_i, 0);
  min_j = std::max(min_j, 0);
  max_i = std::min(max_i, static_cast<int>(master_grid.getSizeInCellsX()));
  max_j = std::min(max_j, static_cast<int>(master_grid.getSizeInCellsY()));

  const unsigned char value = set_unknown_ ? costmap_2d::NO_INFORMATION : costmap_2d::FREE_SPACE;
  for (int j = min_j; j < max_j; ++j)
  {
    for (int i = min_i; i < max_i; ++i)
    {
      double wx, wy;
      master_grid.mapToWorld(i, j, wx, wy);
      if (inside(masked_, wx, wy))
      {
        master_grid.setCost(i, j, value);
      }
    }
  }
}

}  // namespace tiago_move_base_control
