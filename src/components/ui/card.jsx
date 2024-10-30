import * as React from "react"

const Card = React.forwardRef(({ className, children, ...props }, ref) => ( // Added children prop
  <div
    ref={ref}
    className={`rounded-lg border bg-card text-card-foreground shadow-sm ${className}`}
    {...props}
  >
    {children} {/* Added children */}
  </div>
))
Card.displayName = "Card"

const CardHeader = React.forwardRef(({ className, children, ...props }, ref) => ( // Added children prop
  <div
    ref={ref}
    className={`flex flex-col space-y-1.5 p-6 ${className}`}
    {...props}
  >
    {children} {/* Added children */}
  </div>
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef(({ className, children, ...props }, ref) => ( // Added children prop
  <h3
    ref={ref}
    className={`text-2xl font-semibold leading-none tracking-tight ${className}`}
    {...props}
  >
    {children} {/* Added children */}
  </h3>
))
CardTitle.displayName = "CardTitle"

const CardContent = React.forwardRef(({ className, children, ...props }, ref) => ( // Added children prop
  <div ref={ref} className={`p-6 pt-0 ${className}`} {...props}>
    {children} {/* Added children */}
  </div>
))
CardContent.displayName = "CardContent"

export { Card, CardHeader, CardTitle, CardContent }