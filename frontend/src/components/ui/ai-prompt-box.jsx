import React, { createContext, forwardRef, useContext, useEffect, useId, useMemo, useRef, useState } from 'react'
import './ai-prompt-box.css'

const PromptInputContext = createContext(null)

function cn(...classes) {
  return classes.filter(Boolean).join(' ')
}

function usePromptInput() {
  const context = useContext(PromptInputContext)
  if (!context) {
    throw new Error('usePromptInput must be used within a PromptInput')
  }
  return context
}

export const PromptInput = forwardRef(function PromptInput(
  {
    isLoading = false,
    value,
    onValueChange,
    maxHeight = 220,
    onSubmit,
    children,
    className,
    disabled = false,
    ...props
  },
  ref,
) {
  const [internalValue, setInternalValue] = useState(value ?? '')
  const currentValue = value ?? internalValue

  function setValue(nextValue) {
    if (value === undefined) {
      setInternalValue(nextValue)
    }
    onValueChange?.(nextValue)
  }

  const contextValue = useMemo(
    () => ({
      isLoading,
      value: currentValue,
      setValue,
      maxHeight,
      onSubmit,
      disabled,
    }),
    [currentValue, disabled, isLoading, maxHeight, onSubmit],
  )

  return (
    <PromptInputContext.Provider value={contextValue}>
      <div
        ref={ref}
        className={cn('ai-prompt', isLoading && 'ai-prompt-loading', className)}
        role="form"
        aria-label="Prompt input area"
        {...props}
      >
        {children}
      </div>
    </PromptInputContext.Provider>
  )
})

export function PromptInputTextarea({
  disableAutosize = false,
  placeholder,
  className,
  id,
  name = 'prompt',
  onKeyDown,
  rows = 1,
  ...props
}) {
  const { value, setValue, maxHeight, onSubmit, disabled } = usePromptInput()
  const textareaRef = useRef(null)
  const generatedId = useId()
  const textareaId = id ?? `ai-prompt-${generatedId}`

  useEffect(() => {
    if (disableAutosize || !textareaRef.current) return

    textareaRef.current.style.height = 'auto'
    const nextHeight =
      typeof maxHeight === 'number'
        ? `${Math.min(textareaRef.current.scrollHeight, maxHeight)}px`
        : `min(${textareaRef.current.scrollHeight}px, ${maxHeight})`
    textareaRef.current.style.height = nextHeight
  }, [disableAutosize, maxHeight, value])

  function handleKeyDown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      onSubmit?.()
      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSubmit?.()
      return
    }

    onKeyDown?.(event)
  }

  return (
    <textarea
      ref={textareaRef}
      id={textareaId}
      name={name}
      className={cn('ai-prompt-textarea', className)}
      value={value}
      onChange={event => setValue(event.target.value)}
      onKeyDown={handleKeyDown}
      placeholder={placeholder}
      rows={rows}
      disabled={disabled}
      aria-label="Search prompt"
      spellCheck="true"
      autoComplete="off"
      {...props}
    />
  )
}

export function PromptInputActions({ children, className, ...props }) {
  return (
    <div className={cn('ai-prompt-actions', className)} {...props}>
      {children}
    </div>
  )
}

export function PromptInputAction({ tooltip, children, className, side: _side = 'top', ...props }) {
  return (
    <span className={cn('ai-prompt-action', className)} title={typeof tooltip === 'string' ? tooltip : undefined} {...props}>
      {children}
    </span>
  )
}

export function PromptInputSubmit({ children = 'Send', className, ...props }) {
  const { value, isLoading, onSubmit, disabled } = usePromptInput()
  const isDisabled = disabled || isLoading || !value.trim()

  return (
    <button
      type="button"
      className={cn('ai-prompt-submit', className)}
      onClick={onSubmit}
      disabled={isDisabled}
      aria-busy={isLoading}
      {...props}
    >
      {isLoading && <span className="ai-prompt-submit-spinner" aria-hidden="true" />}
      <span>{children}</span>
    </button>
  )
}

export function PromptInputBox({
  onSend,
  isLoading = false,
  placeholder = 'Ask me anything...',
  className,
  value,
  onValueChange,
  disabled = false,
  submitLabel = 'Send',
  leftActions,
  rightActions,
  footer,
  maxHeight = 220,
}) {
  const [internalValue, setInternalValue] = useState(value ?? '')
  const currentValue = value ?? internalValue

  function setValue(nextValue) {
    if (value === undefined) {
      setInternalValue(nextValue)
    }
    onValueChange?.(nextValue)
  }

  function handleSubmit() {
    const message = currentValue.trim()
    if (!message || isLoading || disabled) return

    onSend?.(message, [])
    if (value === undefined) {
      setInternalValue('')
    }
  }

  return (
    <PromptInput
      className={className}
      isLoading={isLoading}
      value={currentValue}
      onValueChange={setValue}
      onSubmit={handleSubmit}
      maxHeight={maxHeight}
      disabled={disabled || isLoading}
    >
      <PromptInputTextarea placeholder={placeholder} rows={2} />
      <div className="ai-prompt-toolbar">
        <PromptInputActions className="ai-prompt-left-actions">{leftActions}</PromptInputActions>
        <PromptInputActions className="ai-prompt-right-actions">
          {rightActions}
          <PromptInputSubmit>{submitLabel}</PromptInputSubmit>
        </PromptInputActions>
      </div>
      {footer && <div className="ai-prompt-footer">{footer}</div>}
    </PromptInput>
  )
}
