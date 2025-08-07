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
 * Check if pipx is available
 */
async function checkPipx() {
  try {
    const result = spawn.sync('pipx', ['--version'], { stdio: 'ignore' });
    return result.status === 0;
  } catch (error) {
    return false;
  }
}

/**
 * Install agent-cli via various methods
 */
async function installAgentCLI(pythonCmd) {
  console.log(chalk.yellow('🔧 Installing agent-cli Python package...'));
  
  // First try pipx (recommended for externally managed environments)
  const hasPipx = await checkPipx();
  if (hasPipx) {
    console.log(chalk.blue('   Trying pipx (recommended for macOS)...'));
    try {
      const result = spawn.sync('pipx', ['install', 'agent-cli'], { stdio: 'inherit' });
      if (result.status === 0) {
        console.log(chalk.green('✅ agent-cli installed successfully with pipx!'));
        return true;
      }
    } catch (error) {
      console.log(chalk.yellow('   pipx installation failed, trying other methods...'));
    }
  }
  
  // Try pip with --user flag (safer for externally managed environments)
  const pipCommands = ['pip3', 'pip'];
  
  for (const pipCmd of pipCommands) {
    try {
      console.log(chalk.blue(`   Trying ${pipCmd} --user...`));
      const result = spawn.sync(pipCmd, ['install', '--user', 'agent-cli'], { stdio: 'inherit' });
      if (result.status === 0) {
        console.log(chalk.green('✅ agent-cli installed successfully!'));
        return true;
      }
    } catch (error) {
      continue;
    }
  }
  
  // Try with python -m pip --user
  try {
    console.log(chalk.blue('   Trying python -m pip --user...'));
    const result = spawn.sync(pythonCmd, ['-m', 'pip', 'install', '--user', 'agent-cli'], { stdio: 'inherit' });
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
  console.log(chalk.yellow('\n📋 Manual installation options (choose one):'));
  
  // Check OS for better instructions
  const platform = os.platform();
  
  if (platform === 'darwin') { // macOS
    console.log(chalk.cyan('\n🍎 For macOS (recommended order):'));
    console.log(chalk.green('   1. Install with pipx (best for macOS):'));
    console.log(chalk.cyan('      brew install pipx'));
    console.log(chalk.cyan('      pipx install agent-cli'));
    console.log(chalk.green('\n   2. Or use virtual environment:'));
    console.log(chalk.cyan('      python3 -m venv ~/agent-cli-env'));
    console.log(chalk.cyan('      source ~/agent-cli-env/bin/activate'));
    console.log(chalk.cyan('      pip install agent-cli'));
    console.log(chalk.green('\n   3. Or install for user only:'));
    console.log(chalk.cyan('      pip install --user agent-cli'));
  } else {
    console.log(chalk.green('\n🐧 General installation:'));
    console.log(chalk.cyan('   pip install agent-cli'));
    console.log(chalk.cyan('   # or'));
    console.log(chalk.cyan('   pipx install agent-cli'));
  }
  
  console.log(chalk.yellow('\n🔗 For more options and troubleshooting:'));
  console.log(chalk.blue('   https://github.com/StamKavid/agent-cli#installation'));
  
  if (platform === 'darwin') {
    console.log(chalk.gray('\n💡 Note: macOS has externally managed Python environments.'));
    console.log(chalk.gray('   pipx is the recommended solution for this issue.'));
  }
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
