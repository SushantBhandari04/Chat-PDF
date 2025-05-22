"use client"

import { PlusCircleIcon } from "lucide-react"
import { Button } from "./ui/button"
import { useRouter } from "next/navigation"

export default function PlaceholderDocument() {
    const router = useRouter();


    const handleClick = () => {
        // Check if user is FREE tier and if they are over the file limit, push to the upgrade page
        router.push("/dashboard/upload");
    }
  
    return (
    <Button onClick={handleClick} className="cursor-pointer flex flex-col items-center w-64 h-80 rounded-xl bg-gray-200 drop-shadow-md text-gray-400">
        <PlusCircleIcon className="h-16 w-16"/>
        <p>Add a document</p>
    </Button>
  )
}
