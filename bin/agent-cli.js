#!/usr/bin/env node

/**
 * Agent CLI - NPM Wrapper
 * 
 * This is an npm wrapper around the Python-based agent-cli tool.
 * It automatically handles Python environment detection and provides
 * a seamless experience for npm users.
 */

const { spawn } = require('cross-spawn');
const chalk = require('chalk');
const path = require('path');
const os = require('os');

const PYTHON_COMMANDS = ['python3', 'python', 'py'];
const AGENT_CLI_MODULE = 'agent_cli';

/**
 * Check if Python is available and has agent-cli installed
 */
async function checkPythonAndAgentCLI() {
  for (const pythonCmd of PYTHON_COMMANDS) {
    try {
      // Check if python command exists
      const pythonCheck = spawn.sync(pythonCmd, ['--version'], { stdio: 'ignore' });
      if (pythonCheck.status !== 0) continue;

      // Check if agent-cli module is available
      const moduleCheck = spawn.sync(pythonCmd, ['-m', AGENT_CLI_MODULE, '--version'], { stdio: 'ignore' });
      if (moduleCheck.status === 0) {
        return pythonCmd;
      }
    } catch (error) {
      continue;
    }
  }
  return null;
}

/**
 * Install agent-cli via pip
 */
async function installAgentCLI(pythonCmd) {
  console.log(chalk.yellow('🔧 Installing agent-cli Python package...'));
  
  const pipCommands = ['pip3', 'pip'];
  
  for (const pipCmd of pipCommands) {
    try {
      const result = spawn.sync(pipCmd, ['install', 'agent-cli'], { stdio: 'inherit' });
      if (result.status === 0) {
        console.log(chalk.green('✅ agent-cli installed successfully!'));
        return true;
      }
    } catch (error) {
      continue;
    }
  }
  
  // Try with python -m pip
  try {
    const result = spawn.sync(pythonCmd, ['-m', 'pip', 'install', 'agent-cli'], { stdio: 'inherit' });
    if (result.status === 0) {
      console.log(chalk.green('✅ agent-cli installed successfully!'));
      return true;
    }
  } catch (error) {
    // Continue to error
  }
  
  return false;
}

/**
 * Show installation instructions
 */
function showInstallationInstructions() {
  console.log(chalk.red('❌ Could not install agent-cli automatically.'));
  console.log(chalk.yellow('\n📋 Manual installation required:'));
  console.log(chalk.cyan('   pip install agent-cli'));
  console.log(chalk.cyan('   # or'));
  console.log(chalk.cyan('   pipx install agent-cli'));
  console.log(chalk.yellow('\n🔗 For more options, visit:'));
  console.log(chalk.blue('   https://github.com/StamKavid/agent-cli#installation'));
}

/**
 * Run agent-cli with the provided arguments
 */
async function runAgentCLI(pythonCmd, args) {
  const result = spawn.sync(pythonCmd, ['-m', AGENT_CLI_MODULE, ...args], { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  
  process.exit(result.status || 0);
}

/**
 * Main function
 */
async function main() {
  const args = process.argv.slice(2);
  
  // Show npm wrapper info for --version
  if (args.includes('--version') || args.includes('-v')) {
    console.log(chalk.blue('📦 agent-cli npm wrapper v0.3.0'));
    console.log(chalk.gray('   Wraps Python agent-cli package'));
  }
  
  // Show help info
  if (args.includes('--help') || args.includes('-h')) {
    console.log(chalk.blue('🤖 Agent CLI - NPM Wrapper'));
    console.log(chalk.gray('   This npm package wraps the Python agent-cli tool.'));
    console.log(chalk.gray('   All commands are passed through to the Python CLI.\n'));
  }
  
  // Check for Python and agent-cli
  console.log(chalk.blue('🔍 Checking Python environment...'));
  let pythonCmd = await checkPythonAndAgentCLI();
  
  if (!pythonCmd) {
    // Try to find Python first
    for (const cmd of PYTHON_COMMANDS) {
      try {
        const check = spawn.sync(cmd, ['--version'], { stdio: 'ignore' });
        if (check.status === 0) {
          pythonCmd = cmd;
          break;
        }
      } catch (error) {
        continue;
      }
    }
    
    if (!pythonCmd) {
      console.log(chalk.red('❌ Python not found. Please install Python 3.10+ first.'));
      console.log(chalk.yellow('🔗 Download Python: https://python.org/downloads/'));
      process.exit(1);
    }
    
    // Python found, but agent-cli not installed
    console.log(chalk.yellow('⚠️  agent-cli Python package not found.'));
    
    const success = await installAgentCLI(pythonCmd);
    if (!success) {
      showInstallationInstructions();
      process.exit(1);
    }
  } else {
    console.log(chalk.green('✅ Python and agent-cli found!'));
  }
  
  // Run the actual agent-cli command
  await runAgentCLI(pythonCmd, args);
}

// Handle unhandled errors
process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('Unhandled Rejection at:'), promise, chalk.red('reason:'), reason);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error(chalk.red('Uncaught Exception:'), error);
  process.exit(1);
});

// Run main function
main().catch((error) => {
  console.error(chalk.red('Error:'), error);
  process.exit(1);
});
