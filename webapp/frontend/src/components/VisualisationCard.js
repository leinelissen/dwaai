import React from "react";
import eindhoven from '../assets/eindhoven.svg';
import eindhovenMuseum from '../assets/eindhoven-museum.svg';

export default class VisualizationCard extends React.Component {
  render() {

    const waveStyle = {
      backgroundColor: `hsl(${ Math.round(this.props.percentage/100 * 255) }, 100%, 50%)`,
    };

    return (
      <section className="postcard">
        <div className="content">
          <div className="analysis">
            {this.props.visualisation.map((height, i) =>
              <div
                key={i}
                style={{ ...waveStyle, height: height * 1.5 }}
              />
            )}
          </div>

          <h2>{ this.props.percentage }% Brabants</h2>
        </div>

        <div className="footer">
          <img src={eindhoven} alt="Eindhoven" />
          <img src={eindhovenMuseum} alt="Eindhoven Museum" />
        </div>
      </section>
    );
  }
}
