import React from "react";
import ReactPlayer from 'react-player'

// reactstrap components
import { Card, CardHeader, CardBody, CardTitle, Row, Col } from "reactstrap";

class Overview extends React.Component {
  render() {
    return (
      <>
        <div className="content">
          <Row>
            <Col md={"12"}>
              <Card>
                <CardHeader className="mb-5">
                  <h5 className="card-category">Open AI Gym</h5>
                  <CardTitle tag="h3">
                    Lunar Lander v2
                  </CardTitle>
                  <CardBody>
                    <Row>
                    <Col md={4}>
                      <ReactPlayer url='https://youtu.be/k-P5EEoKWeI' playsinline={true} width={'100%'} height={'100%'} playing />
                    </Col>
                    <Col md={8}>
                      <p>The task accomplished by this project is to build an AI agent for the game of Lunar Lander defined by openAI gym in Box2D format. Here, a lunar lander needs to land with zero velocity between the flags on a landing pad as shown in the figure. with a constant high reward. This is accomplished by Reinforcement Learning, particularly by applying different Deep Q-learning techniques. This project has explored DQN, Double DQN, and Dueling DQN, to solve the game. We consider the game as solved when the agent starts getting an average reward of 200 over 100 consecutive episodes. Moreover, the performance of different DQN variants are compared to solve the problem.</p>
                      <p>Since, the problem involves high dimensional and continuous state space, standard Q-learning cannot solve this problem unless some amount of discretization is done. Due to this difficulty, Deep Q-Network (DQN) was the choice.</p>
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

export default Overview;
