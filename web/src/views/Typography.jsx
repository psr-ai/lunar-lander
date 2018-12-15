import React from "react";

// reactstrap components
import { Card, CardHeader, CardBody, CardTitle, Row, Col } from "reactstrap";

class Typography extends React.Component {
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
          <Row>
            <Col md="12">
              <Card>
                <CardHeader className="mb-5">
                  <h5 className="card-category">Black Table Heading</h5>
                  <CardTitle tag="h3">
                    Created using Poppins Font Family
                  </CardTitle>
                </CardHeader>
                <CardBody>
                  <div className="typography-line">
                    <h1>
                      <span>Header 1</span>
                      The Life of Black Dashboard React
                    </h1>
                  </div>
                  <div className="typography-line">
                    <h2>
                      <span>Header 2</span>
                      The Life of Black Dashboard React
                    </h2>
                  </div>
                  <div className="typography-line">
                    <h3>
                      <span>Header 3</span>
                      The Life of Black Dashboard React
                    </h3>
                  </div>
                  <div className="typography-line">
                    <h4>
                      <span>Header 4</span>
                      The Life of Black Dashboard React
                    </h4>
                  </div>
                  <div className="typography-line">
                    <h5>
                      <span>Header 5</span>
                      The Life of Black Dashboard React
                    </h5>
                  </div>
                  <div className="typography-line">
                    <h6>
                      <span>Header 6</span>
                      The Life of Black Dashboard React
                    </h6>
                  </div>
                  <div className="typography-line">
                    <p>
                      <span>Paragraph</span>
                      I will be the leader of a company that ends up being worth
                      billions of dollars, because I got the answers. I
                      understand culture. I am the nucleus. I think that’s a
                      responsibility that I have, to push possibilities, to show
                      people, this is the level that things could be at.
                    </p>
                  </div>
                  <div className="typography-line">
                    <span>Quote</span>
                    <blockquote>
                      <p className="blockquote blockquote-primary">
                        "I will be the leader of a company that ends up being
                        worth billions of dollars, because I got the answers. I
                        understand culture. I am the nucleus. I think that’s a
                        responsibility that I have, to push possibilities, to
                        show people, this is the level that things could be at."{" "}
                        <br />
                        <br />
                        <small>- Noaa</small>
                      </p>
                    </blockquote>
                  </div>
                  <div className="typography-line">
                    <span>Muted Text</span>
                    <p className="text-muted">
                      I will be the leader of a company that ends up being worth
                      billions of dollars, because I got the answers...
                    </p>
                  </div>
                  <div className="typography-line">
                    <span>Primary Text</span>
                    <p className="text-primary">
                      I will be the leader of a company that ends up being worth
                      billions of dollars, because I got the answers...
                    </p>
                  </div>
                  <div className="typography-line">
                    <span>Info Text</span>
                    <p className="text-info">
                      I will be the leader of a company that ends up being worth
                      billions of dollars, because I got the answers...
                    </p>
                  </div>
                  <div className="typography-line">
                    <span>Success Text</span>
                    <p className="text-success">
                      I will be the leader of a company that ends up being worth
                      billions of dollars, because I got the answers...
                    </p>
                  </div>
                  <div className="typography-line">
                    <span>Warning Text</span>
                    <p className="text-warning">
                      I will be the leader of a company that ends up being worth
                      billions of dollars, because I got the answers...
                    </p>
                  </div>
                  <div className="typography-line">
                    <span>Danger Text</span>
                    <p className="text-danger">
                      I will be the leader of a company that ends up being worth
                      billions of dollars, because I got the answers...
                    </p>
                  </div>
                  <div className="typography-line">
                    <h2>
                      <span>Small Tag</span>
                      Header with small subtitle <br />
                      <small>Use "small" tag for the headers</small>
                    </h2>
                  </div>
                  <div className="typography-line">
                    <span>Lists</span>
                    <Row>
                      <Col md="3">
                        <h5>Unordered List</h5>
                        <ul>
                          <li>List Item</li>
                          <li>List Item</li>
                          <li className="list-unstyled">
                            <ul>
                              <li>List Item</li>
                              <li>List Item</li>
                              <li>List Item</li>
                            </ul>
                          </li>
                          <li>List Item</li>
                        </ul>
                      </Col>
                      <Col md="3">
                        <h5>Ordered List</h5>
                        <ol>
                          <li>List Item</li>
                          <li>List Item</li>
                          <li>List item</li>
                          <li>List Item</li>
                        </ol>
                      </Col>
                      <Col md="3">
                        <h5>Unstyled List</h5>
                        <ul className="list-unstyled">
                          <li>List Item</li>
                          <li>List Item</li>
                          <li>List item</li>
                          <li>List Item</li>
                        </ul>
                      </Col>
                      <Col md="3">
                        <h5>Inline List</h5>
                        <ul className="list-inline">
                          <li className="list-inline-item">List1</li>
                          <li className="list-inline-item">List2</li>
                          <li className="list-inline-item">List3</li>
                        </ul>
                      </Col>
                    </Row>
                  </div>
                  <div className="typography-line">
                    <span>Code</span>
                    <p>
                      This is <code>.css-class-as-code</code>
                      , an example of an inline code element. Wrap inline code
                      within a <code>{`<code>...</code>`}</code>
                      tag.
                    </p>
                    <pre>
                      1. #This is an example of preformatted text.<br />
                      2. #Here is another line of code
                    </pre>
                  </div>
                </CardBody>
              </Card>
            </Col>
          </Row>
        </div>
      </>
    );
  }
}

export default Typography;
