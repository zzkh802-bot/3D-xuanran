import { defineConfig } from 'vite';

const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1];
const pagesBase = repositoryName?.endsWith('.github.io')
  ? '/'
  : `/${repositoryName ?? '3D-xuanran'}/`;

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? pagesBase : '/',
});
