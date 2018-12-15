/*eslint-disable*/
import React from "react";
// used for making the prop types of this component
import PropTypes from "prop-types";

// reactstrap components
import { Container, Row, Nav, NavItem, NavLink } from "reactstrap";

class Footer extends React.Component {
  render() {
    return (
      <footer className="footer">
        <Container fluid>
          <div className="copyright">
            <Nav>
              <NavItem>
                <NavLink href="javascript:void(0)">PRABHJOT RAI</NavLink>
              </NavItem>
              <NavItem>
                <NavLink href="javascript:void(0)">ABHISHEK BHARANI</NavLink>
              </NavItem>
              <NavItem>
                <NavLink href="javascript:void(0)">AMEY NAIK</NavLink>
              </NavItem>
            </Nav>
          </div>
        </Container>
      </footer>
    );
  }
}

export default Footer;
