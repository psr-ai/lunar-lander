import React from "react";
// react plugin used to create google maps

// reactstrap components
import { Card, CardHeader, CardBody, Row, Col } from "reactstrap";
import env from "assets/img/game_env.png";

class Infrastructure extends React.Component {
  render() {
    return (
      <>
        <div className="content">
          <Row>
            <Col md={"8"}>
              <Card>
                <CardHeader className="mb-5">
                  <h5 className="card-category">Actions</h5>
                  <CardBody>
                    <Row>
                      <Col md={8}>
                        <p>In this game, four discrete actions are available to the playing agent at any time frame. The agent can choose only one action among the given actions at a given time frame.</p>
                      </Col>
                      <Col md={4}>
                        <ol>
                          <li>Do nothing</li>
                          <li>Fire left orientation engine</li>
                          <li>Fire main engine</li>
                          <li>Fire right orientation engine</li>
                        </ol>
                      </Col>
                    </Row>
                  </CardBody>
                </CardHeader>
              </Card>
            </Col>
            <Col md={"4"}>
              <Card>
                <CardHeader className="mb-5">
                  <h5 className="card-category">Terrain</h5>
                  <CardBody>
                    <Row>
                      <Col md={12}>
                        <p>The terrain is a combination of 10 points, and the helipad(landing zone) is fixed between 5th and 6th points towards the center. The values of the height of the landing zone(5th and 6th points on the terrain) are viewport height divided by 4, and the rest of the points are randomly sampled between 0 to H/2 using numpy random and smoothened (averaging 3 continuous points).</p>
                      </Col>
                    </Row>
                  </CardBody>
                </CardHeader>
              </Card>
            </Col>
          </Row>
          <Row>
            <Col md={"4"}>
              <Card>
                <CardHeader className="mb-5">
                  <h5 className="card-category">Game Environment</h5>
                  <CardBody>
                    <img src={env} alt="Game-env" />
                  </CardBody>
                </CardHeader>
              </Card>
            </Col>
            <Col md={"8"}>
              <Card>
                <CardHeader className="mb-5">
                  <h5 className="card-category">State</h5>
                  <CardBody>
                    <Row>
                      <Col md={8}>
                        <p>The state is 8 dimensional values of different parameters of the lunar lander at any given time. The starting state is randomly initialized (the lunar lander takes a step in the world through the ”idle” action) with certain bounds based on the environment.</p>
                      </Col>
                      <Col md={4}>
                        <ol>
                          <li>Position of LunarLander w.r.t X-axis</li>
                          <li>Position of LunarLander w.r.t Y-axis</li>
                          <li>Velocity along X-axis</li>
                          <li>Velocity along Y-axis</li>
                          <li>LunarLander Angle</li>
                          <li>Angular velocity</li>
                          <li>Left leg contacted the surface</li>
                          <li>Right leg contacted the surface</li>
                        </ol>
                      </Col>
                    </Row>
                  </CardBody>
                </CardHeader>
              </Card>
            </Col>
          </Row>
        </div>
      </>
    );
  }
}

export default Infrastructure;
