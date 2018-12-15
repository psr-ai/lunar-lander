import Learning from "views/Learning.jsx";
import Performance from "views/Performance.jsx";
import Map from "views/Map.jsx";
import Notifications from "views/Notifications.jsx";
import Rtl from "views/Rtl.jsx";
import TableList from "views/TableList.jsx";
import Typography from "views/Typography.jsx";
import UserProfile from "views/UserProfile.jsx";

var routes = [
  {
    path: "/performance",
    name: "Performance",
    rtlName: "الرموز",
    icon: "tim-icons icon-atom",
    component: Performance,
    layout: "/lunar-lander"
  },
  {
    path: "/learning",
    name: "Learning",
    rtlName: "لوحة القيادة",
    icon: "tim-icons icon-chart-pie-36",
    component: Learning,
    layout: "/lunar-lander"
  }
];
export default routes;
