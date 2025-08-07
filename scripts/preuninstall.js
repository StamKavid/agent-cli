#!/usr/bin/env node

/**
 * Pre-uninstall script for agent-cli npm package
 */

const chalk = require('chalk');

console.log(chalk.blue('🤖 Agent CLI - NPM Uninstall'));
console.log(chalk.yellow('⚠️  Removing npm wrapper...'));

console.log(chalk.gray('\n📝 Note: This only removes the npm wrapper.'));
console.log(chalk.gray('   The Python agent-cli package will remain installed.'));

console.log(chalk.yellow('\n🗑️  To completely remove agent-cli:'));
console.log(chalk.cyan('   pip uninstall agent-cli'));

console.log(chalk.blue('\n👋 Thanks for using agent-cli!'));
