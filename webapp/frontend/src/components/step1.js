import React from "react";

const intro = {
  en: 'Pronounce the sentence above in your best Brabants accent and find out!',
  nl: 'Spreek de zin hierboven uit in je beste Brabantse accent en kom erachter!',
  br: 'Kende gij de zin hierboven uitsprèke? Dan kenne wij oe vertelle of gij enne Brabander zijt',
}

const cta = {
  en: 'Start now!',
  nl: 'Begin nu!',
  br: 'Gauw beginnen!',
}

export default function Step1(props) {
  return (
    <div>
      <h1 className="style-font">
        Bende gij een Brabander?
      </h1>
      <p>{intro[props.language]}</p>
      <button className="style-font" onClick={() => props.setStep(2)}>{cta[props.language]}</button>
    </div>
  );
}
