#!/usr/bin/env node

/**
 * Inspired by https://github.com/IronCoreLabs/recrypt-node-binding
 * ==================================
 *
 * This script is responsible for compiling and building the NPM release bundle for this repo. The following steps are taken:
 *
 * + Clean up any existing Rust builds by running `cargo clean`.
 * + Run `cargo update` to make sure all dependencies are available.
 * + Compile rust code into index.node file.
 * + Run unit tests to ensure the library is in good shape for publishing.
 * + Move all expected content into a `dist` directory.
 * + Generate a binary distrubtion in `bin-package`.
 * + Do a dry run of npm publishing via irish-pub or perform an actual publish step if `--publish` option is provided.
 */

const fs    = require("fs");
const path  = require("path");
const shell = require("shelljs");

const distPath = "./dist";

// Fail this script if any of these commands fail
shell.set("-e");

// Ensure that our directory is set to the root of the repo
const rootDirectory = path.dirname(process.argv[1]);
shell.cd(rootDirectory);

const arg = process.argv.slice(2)[0];
switch (arg) {
  case "--all":
    buildRust();
    buildTs();
    break;

  case "--rust":
    buildRust();
    break;

  case "--typescript":
    buildTs();
    break;

  case "--package-rust":
    buildRust();
    packageRust();
    break;

  case "--npm-publish":
    buildTs();
    npmPublish();
    break;

  default:
    shell.echo('No arg provided, doing nothing...');
    break;
}

function buildRust() {
  shell.echo('BUILDING RUST...');

  // Cleanup the previous build, if it exists
  shell.rm("-rf", "./bin-package");
  shell.rm("-rf", "./build");
  
  // Cleanup any previous Rust builds, update deps, and compile
  shell.exec("npm ci --ignore-scripts");
  shell.exec("npm run clean");
  shell.pushd("./native");
  shell.exec("cargo update");
  shell.popd();
  shell.exec("npm run compile");

  shell.echo('BUILDING RUST COMPLETE...');
}

function packageRust() {
  shell.echo('PACKAGING RUST...');

  shell.mkdir("./bin-package");
  shell.cp("./native/index.node", "./bin-package");
  shell.exec("npm run package");
  var tgz = shell.exec("find ./build -name *.tar.gz");
  shell.cp(tgz, "./bin-package/");

  shell.echo('PACKAGING RUST COMPLETE...');
}

function buildTs() {
  shell.echo('BUILDING TS...');

  // Cleanup the previous build, if it exists
  shell.rm("-rf", distPath);

  shell.exec("npm ci --ignore-scripts");
  shell.mkdir(distPath);

  shell.exec("npx tsc");
  shell.cp("-r", ["lib/bindings"], distPath);
  shell.mv([`${distPath}/bindings/native.prod.js`], [`${distPath}/bindings/native.js`]);

  shell.echo('BUILDING TS COMPLETE...');
}

async function npmPublish() {
  shell.echo('PUBLISHING ON NPM...');

  shell.cp("-r", ["package.json", "README.md", "../../LICENSE"], distPath);

  // Add a NPM install script to the package.json that we push to NPM so that when consumers pull it down it
  // runs the expected node-pre-gyp step.
  const npmPackageJson = require(`${distPath}/package.json`);
  npmPackageJson.scripts.install = "node-pre-gyp install";
  await fs.promises.writeFile(`${distPath}/package.json`, JSON.stringify(npmPackageJson, null, 2));

  shell.exec(`npm publish ${distPath} --access public`);

  shell.echo('PUBLISHING ON NPM COMPLETE...');
}
