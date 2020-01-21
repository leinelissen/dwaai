require('dotenv-defaults').config()

const path = require('path');
const express = require('express');
const app = express();
const port = process.env.PORT || 5000;
const multer  = require('multer');
const exec = require("child_process").exec;

// Create a DiskStorage system so that we can modify the filenames
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, path.resolve(__dirname, 'uploads')),
  filename: (req, file, cb) => cb(null, Date.now() + '.wav'),
});

// Then initialise the multer middleware using the DiskStorage system
const upload = multer({ storage: storage })

// console.log that your server is up and running
app.listen(port, () => console.log(`Listening on port ${port}`));

// Declare the route that receives the audio file
app.post('/recording', upload.single('audio'), function (req, res) {
  // Log the file coming in
  console.log(req.file);

  if (process.env.DEBUG === 'true') {
    return res.send({ result: Math.round(Math.random() * 99) });
  }

  // Spawn python process
  const scriptPath = path.resolve(__dirname, '..', 'mfcc', process.env.MODEL_SCRIPT_NAME);
  const pythonProcess = exec(`${process.env.PYTHON_BIN} "${scriptPath}" "${req.file.path}"`, (error, stdout, stderr) => {
    if (error || stderr) {
      // Log error message to console
      console.log('Received error from python: ', error, stderr);

      // Return error to front-end
      return res.sendStatus(500);
    }

    console.log('Received data from python: ', stdout);

    // Process response
    const [ randomForest, gradientBoosting, convolutionalNeuralNetwork ] = stdout.split(',');

    // Defer response
    return res.send({ 
      randomForest: parseFloat(randomForest),
      gradientBoosting: parseFloat(gradientBoosting),
      convolutionalNeuralNetwork: parseFloat(convolutionalNeuralNetwork),
     });
  });
});