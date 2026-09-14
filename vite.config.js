import { defineConfig } from 'vite';
import { readFileSync } from 'node:fs';

const dataFiles = [
  'userprofile.csv',
  'geoplaces2.csv',
  'rating_final.csv',
  'usercuisine.csv',
  'userpayment.csv',
  'chefmozcuisine.csv',
  'chefmozaccepts.csv',
  'chefmozparking.csv',
  'chefmozhours4.csv'
];

export default defineConfig({
  plugins: [{
    name: 'bundle-recommender-data',
    apply: 'build',
    buildStart() {
      for (const fileName of dataFiles) {
        this.emitFile({ type: 'asset', fileName: `dataset/${fileName}`, source: readFileSync(`dataset/${fileName}`) });
      }
    }
  }]
});
