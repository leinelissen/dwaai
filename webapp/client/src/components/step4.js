import React, { useEffect, useRef } from "react";
import ReactToPrint from 'react-to-print';
import VisualisationCard from './VisualisationCard';

export default function Step4(props) {
  const componentRef = useRef();

  const renderResult = (result) => {
    if(result >= 80){
      return <span>
                <span role="img" aria-label="party popper">🎉</span>
                  Yes, you are!
                <span role="img" aria-label="party popper">🎉</span>
            </span>;
    }else if(result>=30){
      return <span>
                <span role="img" aria-label="clapping hands">👏</span>
                  That's quite okay!
                <span role="img" aria-label="clapping hands">👏</span>
            </span>;
    }else{
      return <span>
                <span role="img" aria-label="raising hands">🙌</span>
                  Not yet, but getting there!
                <span role="img" aria-label="raising hands">🙌</span>
            </span>;
    }
  }

  return (
    <div>
      <h1 className="style-font">{ renderResult(props.recordingResult) }</h1>
      <h2>Your score is { props.recordingResult }%</h2>
      <button className="style-font" onClick={() => props.setStep(3)}>&#8634; Retry</button>

      <ReactToPrint
        trigger={() => <button className="style-font">&#128438; Print</button>}
        content={() => componentRef.current}
        onAfterPrint={ () => props.setStep(1) }
        bodyClass="A5Landscape"
        pageStyle='@page { size: A5 landscape; margin: 0mm; } @media print { body { -webkit-print-color-adjust: exact; padding: 40px !important; } }'
      />
      <div style={{ display: "none" }}>
        <VisualisationCard ref={componentRef} percentage={props.recordingResult} visualisation={props.visualisationResult} />
      </div>
    </div>
  );
}
