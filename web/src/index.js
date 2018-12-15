import React from "react";
import ReactDOM from "react-dom";
import { createBrowserHistory } from "history";
import { Router, Route, Switch, Redirect } from "react-router-dom";

import AdminLayout from "layouts/Admin/Admin.jsx";
import RTLLayout from "layouts/RTL/RTL.jsx";

import "assets/scss/black-dashboard-react.scss";
import "assets/demo/demo.css";
import "assets/css/nucleo-icons.css";

const hist = createBrowserHistory();

ReactDOM.render(
  <Router history={hist}>
    <Switch>
      <Redirect exact from="/lunar-lander" to="/lunar-lander/performance" />
      <Route path="/lunar-lander" render={props => <AdminLayout {...props} />} />
      <Redirect from="/" to="/lunar-lander/performance" />
    </Switch>
  </Router>,
  document.getElementById("root")
);
