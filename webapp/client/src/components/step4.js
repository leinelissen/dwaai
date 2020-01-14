import React, { useRef } from "react";
import ReactToPrint from 'react-to-print';
import VisualisationCard from './VisualisationCard';

const resultYes = {
  en: 'Yes, you are!',
  nl: 'Ja, je ben het!',
  br: 'Ja, ge bent!',
}

const resultAlmost = {
  en: 'That\'s quite okay!',
  nl: 'Dat klinkt al goed!',
  br: 'Denk \'t bijna wel of nie?',
}

const resultNo = {
  en: 'Not yet, but getting there!',
  nl: 'Nog niet, maar gaat de goede kant op!',
  br: 'Nog nie, efkes oefenen nog!',
}

const score = {
  en: 'Your score is',
  nl: 'Je score is',
  br: 'Je score is',
}

const retry = {
  en: 'Retry',
  nl: 'Opnieuw',
  br: 'Opnieuw',
}

export default function Step4(props) {
  const componentRef = useRef();
  const { recordingResult: { gradientBoosting } } = props;
  const modelResult = Math.round(gradientBoosting * 100);

  const getHeadline = (result) => {
    if(result >= 80){
      return <span>{resultYes[props.language]}</span>;
    }else if(result>=30){
      return <span>{resultAlmost[props.language]}</span>;
    }else{
      return <span>{resultNo[props.language]}</span>;
    }
  }

  const getEmoji = (result) => {
    if(result >= 80){
      return <span role="img" aria-label="party popper">🎉</span>;
    }else if(result>=30){
      return <span role="img" aria-label="clapping hands">👏</span>;
    }else{
      return <span role="img" aria-label="raising hands">🙌</span>;
    }
  }

  return (
    <div>
      <h1 className="style-font">{ getHeadline(modelResult) }</h1>
      <h2>{ getEmoji(modelResult) } {score[props.language]} { modelResult }%</h2>
      <button className="style-font" onClick={() => props.setStep(3)}><img src="/undo-alt-regular.svg" alt="Retry" /> {retry[props.language]}</button>

      <ReactToPrint
        trigger={() => <button className="style-font"><img src="/print-regular.svg" alt="Retry" /> Print</button>}
        content={() => componentRef.current}
        onAfterPrint={ () => props.setStep(1) }
        bodyClass="A5Landscape"
        pageStyle='@page { size: A5 landscape; margin: 0mm; } @media print { body { -webkit-print-color-adjust: exact; padding: 40px !important; } }'
      />
      <div style={{ display: "none" }}>
        <VisualisationCard ref={componentRef} percentage={modelResult} visualisation={props.visualisationResult} />
      </div>
    </div>
  );
}
