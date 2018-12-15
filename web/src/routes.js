import Learning from "views/Learning.jsx";
import Performance from "views/Performance.jsx";
import Infrastructure from "views/Infrastructure.jsx";
import Notifications from "views/Notifications.jsx";
import Rtl from "views/Rtl.jsx";
import TableList from "views/TableList.jsx";
import Overview from "views/Overview.jsx";
import UserProfile from "views/UserProfile.jsx";

var routes = [
  {
    path: "/overview",
    name: "Overview",
    icon: "tim-icons icon-align-center",
    component: Overview,
    layout: "/lunar-lander"
  },
  {
    path: "/infra",
    name: "Infrastructure",
    icon: "tim-icons icon-puzzle-10",
    component: Infrastructure,
    layout: "/lunar-lander"
  },
  {
    path: "/learning",
    name: "Learning",
    icon: "tim-icons icon-chart-pie-36",
    component: Learning,
    layout: "/lunar-lander"
  },
  {
    path: "/performance",
    name: "Performance",
    icon: "tim-icons icon-atom",
    component: Performance,
    layout: "/lunar-lander"
  }
];
export default routes;
