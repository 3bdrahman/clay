/// <reference types="vite/client" />

interface ErrorConstructor {
  captureStackTrace(error: Error, constructorOpt?: Function): void;
}
