# Webapp
The application consists of a front-end, powered by React and initialised with Create React App, as well as a back-end, which is powered by Express. They provide for a recording solution that takes users to the flow of recording an audio sampled, which is then passed over to Python for analysis, after which the result is routed back to the front-end via our back-end application.

## Prerequisites
Since both applications are based on [NodeJS](https://nodejs.org/en/), you need to install it first, before you can start the applications. We recommend the v12 LTS version.

## Setting up the front-end and back-end
Go into each folder and run the following commands:
```
npm install
npm start
```
A browser window will open for (http://localhost:3000)[http://localhost:3000], which contains the front-end application screen

## Setting environment variables
In case environment variables need to be set for any of the apps, please follow these instructions:
### Back-end
Environment variables should be set in the `.env` file. Follow `NAME=value` syntax. Available environment variables can be found in the `.env.defaults` file. You should copy it if you are going to make any adjustments.
```
cp .env.defaults .env
```
### Front-end
The front-end variables work similarly, but the file is named `.env.local`, while the defaults file is named `.env`. Copy it when making adjustments:
```
cp .env .env.local
```
Please note that Data Foundry variables are recommended when doing a deployment.