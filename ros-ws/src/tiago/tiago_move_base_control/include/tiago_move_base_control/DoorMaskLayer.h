#pragma once

#include <costmap_2d/layer.h>
#include <geometry_msgs/PolygonStamped.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <mutex>
#include <string>
#include <vector>

namespace tiago_move_base_control
{

// Clears the cells inside a polygon (the door leaf's swing sector) while enabled
class DoorMaskLayer : public costmap_2d::Layer
{
public:
  DoorMaskLayer();

  void onInitialize() override;

  void updateBounds(double robot_x, double robot_y, double robot_yaw,
                    double* min_x, double* min_y, double* max_x, double* max_y) override;

  void updateCosts(costmap_2d::Costmap2D& master_grid,
                   int min_i, int min_j, int max_i, int max_j) override;

  void reset() override {}

private:
  struct Point
  {
    double x;
    double y;
  };

  void polygonCb(const geometry_msgs::PolygonStampedConstPtr& msg);
  void enabledCb(const std_msgs::BoolConstPtr& msg);
  // The polygon in the costmap's frame; false while TF cannot provide it.
  bool polygonInCostmapFrame(std::vector<Point>& points);
  static bool inside(const std::vector<Point>& polygon, double x, double y);

  std::string polygon_topic_;
  std::string enabled_topic_;
  bool set_unknown_;

  ros::Subscriber polygon_sub_;
  ros::Subscriber enabled_sub_;

  std::mutex mutex_;
  geometry_msgs::PolygonStamped polygon_;
  bool have_polygon_;
  bool enabled_;
  // Area last masked, repainted once after the mask is disabled or moved.
  bool repaint_;
  double repaint_min_x_, repaint_min_y_, repaint_max_x_, repaint_max_y_;
  std::vector<Point> masked_;
};

}  // namespace tiago_move_base_control
