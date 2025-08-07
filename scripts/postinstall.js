#!/usr/bin/env node

/**
 * Post-install script for agent-cli npm package
 */

const chalk = require('chalk');
const { spawn } = require('cross-spawn');

console.log(chalk.blue('🤖 Agent CLI - NPM Installation Complete!'));
console.log(chalk.green('✅ NPM wrapper installed successfully.'));

console.log(chalk.yellow('\n📋 Next Steps:'));
console.log(chalk.cyan('   1. The npm wrapper will automatically install the Python package on first use'));
console.log(chalk.cyan('   2. Or install manually: pip install agent-cli'));
console.log(chalk.cyan('   3. Test installation: agent-cli --help'));

console.log(chalk.yellow('\n🚀 Quick Start:'));
console.log(chalk.cyan('   agent-cli my-awesome-project'));
console.log(chalk.cyan('   cd my-awesome-project'));
console.log(chalk.cyan('   pip install -e .'));

console.log(chalk.yellow('\n🔗 Documentation:'));
console.log(chalk.blue('   https://github.com/StamKavid/agent-cli'));

console.log(chalk.green('\n🎉 Happy agent building!'));
