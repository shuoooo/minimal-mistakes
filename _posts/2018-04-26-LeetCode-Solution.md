---
title: "My LeetCode Solutions"
---

This document tracks the LeetCode problems that I have finished, as well as my solutions. I use python 3.4 in the following solutions, though the versions submitted to LeetCode platform could be slightly modified.

# Table of contents
[1. Two Sum](#1) <br/>
[2. Add Two Numbers](#2) <br/>
[3. Longest Substring Without Repeating Characters](#3) <br/>
[7. Reverse Integer](#7) <br/>
[9. Palindrome Number](#9) <br/>
[13. Roman to Integer](#13) <br/>
[14. Longest Common Prefix](#14) <br/>
[20. Valid Parentheses](#20) <br/>
[21. Merge Two Sorted Lists](#21) <br/>
[26. Remove Duplicates from Sorted Array](#26) <br/>
[27. Remove Element](#27) <br/>
[28. Implement strStr()](#28) <br/>
[35. Search Insert Position](#35) <br/>
[38. Count and Say](#38) <br/>
[53. Maximum Subarray](#53) <br/>
[58. Length of Last Word](#58) <br/>
[66. Plus One](#66) <br/>
[67. Add Binary](#67) <br/>
[69. Sqrt(x)](#69) <br/>
[70. Climbing Stairs](#70) <br/>
[83. Remove Duplicates from Sorted List](#83) <br/>
[88. Merge Sorted Array](#88) <br/>
[100. Same Tree](#100) <br/>
[101. Symmetric Tree](#101) <br/>
[104. Maximum Depth of Binary Tree](#104) <br/>
[118. Pascal\'s Triangle](#118) <br/>
[119. Pascal\'s Triangle II](#119) <br/>
[167. Two Sum II - Input array is sorted](#167) <br/>
[169. Majority Element](#169) <br/>
[242. Valid Anagram](#242) <br/>
[278. First Bad Version](#278) <br/>
[349. Intersection of Two Arrays](#349) <br/>
[350. Intersection of Two Arrays II](#350) <br/>
[687. Longest Univalue Path](#687) <br/>
[698. Partition to K Equal Sum Subsets](#698) <br/>

--------------------------------------------------------

## 1. Two Sum <a name="1"></a>

**Solution:** use hash table, with only one path through the array.
For any item `nums[i]` in the array, the hash table stories `target-nums[i]:i`.

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        temp = {}
        for i in range(len(nums)):
            if nums[i] in temp:
                if i < temp[nums[i]]:
                    return [i,temp[nums[i]]]
                else:
                    return [temp[nums[i]],i]
            temp[target-nums[i]] = i
```
--------------------------------------------------------

## 2. Add Two Numbers <a name="2"></a>

**Solution:** simple problem, no particular idea.

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = ListNode(-1)
        current = head
        adder = 0
        while(l1 != None and l2 !=None):
            temp = l1.val + l2.val + adder
            current.next = ListNode(temp % 10)
            adder = int(temp / 10)
            current = current.next
            l1 = l1.next
            l2 = l2.next

        if l1==None and l2==None:
            if adder > 0:
                current.next = ListNode(adder)
            return head.next

        while(l2 != None):
            if adder == 0:
                current.next = l2
                break
            current.next = ListNode((l2.val + adder)%10)
            adder = int((l2.val + adder)/10)
            current = current.next
            l2 = l2.next

        while(l1 != None):
            if adder ==0:
                current.next = l1
                break
            current.next = ListNode((l1.val + adder)%10)
            adder = int((l1.val + adder)/10)
            current = current.next
            l1 = l1.next

        if adder > 0:
            current.next = ListNode(adder)
        return head.next
```
--------------------------------------------------------

## 3. Longest Substring Without Repeating Characters <a name="3"></a>

**Solution 1:** Sliding window

```python
class Solution(object):
    def lengthOfLongestSubstring(self,s):
        """
        :type s: str
        :rtype: int
        """
        max_len, l, i, j = 0, len(s), 0, 0        
        while j < l:
            if s[j] not in s[i:j]:
                if j-i+1 > max_len:
                    max_len = j-i+1
                j += 1
            else:
                k = s[i:j].index(s[j])
                i = i + k +1
        return max_len
```

**Solution 2:** Sliding window with hash table

```python
class Solution(object):
    def lengthOfLongestSubstring(self,s):
        """
        :type s: str
        :rtype: int
        """
        max_len, i, index = 0, 0, {}
        for j in range(len(s)):
            if s[j] not in index or index[s[j]] < i:
                max_len = max(max_len, j-i+1)
            else:
                i = index[s[j]] + 1
            index[s[j]] = j
        return max_len
```

--------------------------------------------------------

## 7. Reverse Integer <a name="7"></a>

**Solution:** reverse using python `reversed()` function, and check the integer range.
```python
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = x > 0
        temp = list(str(abs(x)))
        temp.reverse()
        temp = int("".join(temp))

        if sign:
            if temp > 2**31-1:
                return 0
            return temp
        else:
            if -temp < -2**31:
                return 0
            return -temp
```

--------------------------------------------------------

## 9. Palindrome Number <a name="9"></a>

**Solution:** Simple!

```python
class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x<0:
        	return False

        original_x = x
        inverse_x = 0
        while(x != 0):
        	tail = x % 10
        	x = int(x/10)
        	inverse_x = inverse_x*10 + tail
        if inverse_x == original_x:
        	return True
        else:
        	return False
```

--------------------------------------------------------

## 13. Roman to Integer <a name="13"></a>

**Solution:** if higher level Roman before lower level one, then add; otherwise, subtract
```python
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """

        dict = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D': 500, 'M':1000}
        value = 0
        for i in range(len(s)-1):
        	if dict[s[i]] >= dict[s[i+1]]:
        		value += dict[s[i]]
        	else:
        		value -= dict[s[i]]
        value += dict[s[-1]]
        return value
```

--------------------------------------------------------

## 14. Longest Common Prefix <a name="14"></a>

**Solution:** One simplest solution is to first sort these strings, and compare the first and the last string. Time complexity would be `O(nlogn)`.

```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs)==0:
        	return ""
        strs_sorted = sorted(strs)
        x = strs_sorted[0]
        y = strs_sorted[-1]
        output = ""

        for i in range(len(x)):
        	if x[i] == y[i]:
        		output = output + x[i]
        	else:
        		break
        return output
```

--------------------------------------------------------

## 20. Valid Parentheses <a name="20"></a>

**Solution:** Use stack, simple!

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        dict = {'(':1, ')':-1, '[':2, ']':-2, '{':3, '}':-3}
        stack = []
        for i in s:
            temp = dict[i]
            if len(stack) == 0:
                stack.append(temp)
            else:
                if temp + stack[-1] == 0:
                    stack.pop()
                else:
                    stack.append(temp)
        if len(stack)==0:
            return True
        else:
            return False
```

--------------------------------------------------------

## 21. Merge Two Sorted Lists <a name="21"></a>

**Solution:** Simple merge, I try not to use recursion to aviod the time overhead of calling functions for multiple times.

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        r = ListNode(None)
        head = r
        p1, p2 = l1, l2

        while(p1 != None and p2 != None):
        	if p1.val < p2.val:
        		r.next = ListNode(p1.val)
        		r = r.next
        		p1 = p1.next
        	else:
        		r.next = ListNode(p2.val)
        		r = r.next
        		p2 = p2.next

        if p1 != None:
        	r.next = p1
        if p2 != None:
        	r.next = p2
        return head.next
```

--------------------------------------------------------

## 26. Remove Duplicates from Sorted array <a name="26"></a>

**Solution:** very simple.

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = 0
        if len(nums)>=1:
        	l = 1
        for i in range(1,len(nums)):
        	if nums[i] != nums[i-1]:
        		nums[l] = nums[i]
        		l += 1
        return l
```

--------------------------------------------------------

## 27. Remove Element <a name="27"></a>

**Solution:** simple idea. Note that if the element to remove is rare, then swap with elements at the end of the array.

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """

        l = 0
        for i in range(len(nums)):
        	if nums[i] != val:
        		nums[l] = nums[i]
        		l += 1
        return l
```

--------------------------------------------------------

## 28. Implement strStr() <a name="28"></a>

**Solution:** simple solution, sliding to search.

```python
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if haystack ==0:
        	return 1

        h = len(haystack)

        n = len(needle)
        for i in range(1+h-n):
        	if haystack[i:i+n] == needle:
        		return i
        return -1
```

--------------------------------------------------------

## 35. Search Insert Position <a name="35"></a>

**Solution:** binary search.
```python
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums)-1

        while left<right:        
            mid = int((left+right)/2)
            if target == nums[mid]:
                return mid
            if target < nums[mid]:
                right = mid-1
            elif target > nums[mid]:
                left = mid+1

        if target<=nums[left]:
            return left
        if target>nums[left]:
            return left+1
```

--------------------------------------------------------

## 38. Count and Say <a name="38"></a>

**Solution:** go by definition.
```python
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n ==1:
        	return '1'

        prev = '1'
        for i in range(1,n):
        	result = ""
        	j = 1
        	count = 1

        	while(j < len(prev)):
        		if prev[j] == prev[j-1]:
        			count += 1
        		else:
        			result = result + str(count) + prev[j-1]
        			count = 1
        		j += 1
        	result = result + str(count) + prev[j-1]
        	prev = result

        return prev
```
--------------------------------------------------------

## 53. Maximum Subarray <a name="53"></a>
This is a very interesting problem. Various solutions can be applied. Below, I provide several simple and efficient solutions.

**Solution 1:** One natural idea is dynamic programming, i.e., searching from left to right and keeping track of the maximum sum before current search index.

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        prev = nums[0]
        max_sum = nums[0]
        for i in range(1,len(nums)):
        	if prev > 0:
        		prev = nums[i] + prev
        	else:
        		prev = nums[i]
        	max_sum = max(max_sum, prev)
        return max_sum
```

**Solution 2:** keeping track of the minimum sum, and using `res` to record the current sum minus the minimum sum. The largest `res` is the maximum subarray.

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        sum, min, res = 0, 0, nums[0]
        for i in nums:
        	sum += i
        	if sum - min > res:
        		res = sum - min
        	if sum < min:
        		min = sum
        return res
```

**Solution 3:** more simple solutio would be: if current sum is small than 0, than get rid of it.

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        sum, max = 0, nums[0]

        for i in nums:
            sum += i
            if sum > max:
                max = sum
            if sum < 0:
                sum = 0
        return max
```

--------------------------------------------------------

## 58. Length of Last Word <a name="58"></a>

**Solution:** search from right to left.

```python
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        right = len(s) - 1
        while right>=0:
            if s[right] != ' ':
                break
            else:
                right -= 1

        left = right
        while left>=0:
            if s[left] != ' ':
                left -= 1
            else:
                break

        return right-left
```

--------------------------------------------------------

## 66. Plus One <a name="66"></a>

**Solution:** simple!

```python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
		adder = 1
		x = len(digits)-1
		while x>=0:
			adder, digits[x]= (digits[x]+adder)>10, (digits+adder)%10
			if adder ==0:
				break
			x -= 1
		if adder ==1:
			return [1].append(digits)
```

--------------------------------------------------------

## 67. Add Binary <a name="67"></a>

**Solution:** simple! Aviod using python package or built-in functions.

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if len(a) > len(b):
            a, b = b, a

        x = ''.join(reversed(a))
        y = ''.join(reversed(b))
        l1 = len(x)
        l2 = len(y) # x is the shorter one

        adder = 0
        result = ''

        for i in range(l1):
            temp = int(x[i]) + int(y[i]) + adder
            if temp < 2:
                result = result + str(temp)
                adder = 0
            else:
                result = result + str(temp-2)
                adder = 1

        for i in range(l1, l2):
            temp = int(y[i]) + adder
            if temp<2:
                result = result + str(temp)
                adder = 0
            else:
                result = result + str(temp-2)
                adder = 1

        if adder ==1 :
            result = result + '1'

        return ''.join(reversed(result))

print(Solution().addBinary('1010','1011'))

```

--------------------------------------------------------

## 69. Sqrt(x) <a name="69"></a>

**Solution 1:** binary search.
```python
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        left = 0
        right = x

        while(right > left):
        	mid = left + (right - left + 1)/2
        	if x/mid >= mid:
        		left = mid
        	else:
        		right = mid - 1
        return right
```

**Solution 2:** Newton's method.
```python
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x ==0:
        	return 0

        r_prev = x
        while True:
        	r = r_prev - (r_prev*r_prev - x)/(2.0*r_prev)
        	if r_prev - r <1:
        		return int(r)
        	r_prev = r
```

--------------------------------------------------------

## 70. Climbing Stairs <a name="70"></a>

**Solution:** dynamic programming

```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n<3:
        	return n

        result = [0,1,2]

        for i in range(3,n+1):
        	result.append(result[i-1]+result[i-2])
        return result[-1]
```

--------------------------------------------------------

# 83. Remove Duplicates from Sorted Lists <a name="83"></a>

**Solution:** simple question, tips: check if `head == None`, and also remember to remove duplicates at the end of the linked list.

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None
        else:
            h = head
            r = head.next
            prev = head.val
        while(r != None):
            if r.val != prev:
                h.next = r
                h = r
                prev = r.val
            r = r.next
        h.next = None

        return head

```

--------------------------------------------------------

## 88. Merge Sorted Array <a name="88"></a>

**Solution:** In-place modification is required by the problem. So, place element large to small, from nums1[m+n-1] to num1[0], `O(m+n)` time complexity.

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
        return None
```

--------------------------------------------------------

## 100. Same Tree <a name="100"></a>
**Solution 1:** Non-recursive breadth first search implementation.

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        list1, list2  = [p], [q]
        while len(list1)!=0 and len(list2)!=0:
            a = list1.pop(0)
            b = list2.pop(0)
            if a != None and b != None:
                if a.val != b.val:
                    return False
                list1.append(a.left)
                list1.append(a.right)
                list2.append(b.left)
                list2.append(b.right)
            elif a != b:
                return False
        if len(list1) != len(list2):
            return False
        return True
```

**Solution 2:** Recursive implementation.

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            a = isSameTree(p.left, q.left)
            b = isSameTree(p.right, q.right)
            return a and b
        else:
            return False
```

--------------------------------------------------------

## 101. Symmetric Tree <a name="101"></a>

**Solution 1:** Non-recursive, use queue
```python
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        q = [root.left, root.right]
        while len(q)!=0:
            l = q.pop(0)
            r = q.pop(0)
            if l == None and r == None:
                continue
            if l == None or r == None:
                return False
            if l.val != r.val:
                return False
            q.append(l.left)
            q.append(r.right)
            q.append(l.right)
            q.append(r.left)
        return True
```

**Solution 2:** Recursive
```python
class Solution(object):
    def isSymmetricEqual(self, l, r):
        if l == None and r == None:
            return True
        if l == None or r == None:
            return False
        if l.val != r.val:
            return False
        x = self.isSymmetricEqual(l.left, r.right)
        y = self.isSymmetricEqual(l.right, r.left)
        return x and y

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        return self.isSymmetricEqual(root.left, root.right)
```

--------------------------------------------------------

## 104. Maximum Depth of Binary Tree <a name="104"></a>

**Solution:** recursive method
```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        return 1+ max(self.maxDepth(root.left), self.maxDepth(root.right))
```

--------------------------------------------------------

## 118. Pascal\'s Triangle <a name="118"></a>

**Solution:** dynamic programming

```python
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        result = [[1]]
        if numRows==0:
        	return []
        if numRows==1:
        	result

        for i in range(2,numRows+1):
        	temp = [1]
        	for j in range(1,i-1):
        		temp.append(result[-1][j-1] + result[-1][j])
        	temp.append(1)
        	result.append(temp)
        return result
```

--------------------------------------------------------

## 119. Pascal\'s Triangle II <a name="119"></a>

**Solution:** Use the Pascal's triangle fomulation.

```python
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        result= [1]
        temp = 1
        for i in range(1, rowIndex+1):
            temp = temp * (rowIndex-i+1) / i
            result.append(int(temp))
        return result
```

--------------------------------------------------------

## 167. Two Sum II - Input array is sorted <a name="167"></a>

**Solution 1:** One solution is to still use the hash table, which does not take advantage of the sorted property.

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """

        x = {}
        for i in range(len(numbers)):
        	temp = numbers[i]
        	if temp in x:
        		return [min(i,x[temp])+1,max(i,x[temp])+1]
        	else:
        		x[target-temp] = i
```

**Solution 2:** Since the array is sorted, we can use binary search to find the solution, which could be more efficient.

```python
class Solution(object):
    def update_right(self,left,right,numbers,x):
        # find the largest item that is <= x
        if numbers[right] <= x:
            return right
        while(right > left):
            mid = int((right+left)/2)
            if numbers[mid] > x:
                right = mid
            else:
                left = mid+1
        if numbers[left] == x:
            return left
        else:
            return left-1

    def update_left(self, left, right, numbers, x):
        # find the smallest item that is >= x
        if numbers[left] >= x:
            return left
        while(right > left):
            mid = int((right+left)/2)
            if numbers[mid] < x:
                left = mid+1
            else:
                right = mid
        return left

    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        left = 0
        right = len(numbers)-1

        if numbers[left]+numbers[right] == target:
            return (left+1, right+1)

        while(True):
            right = self.update_right(left, right, numbers,target - numbers[left])
            if numbers[left]+numbers[right] == target:
                return (left+1, right+1)            
            left = self.update_left(left, right, numbers, target - numbers[right])
            if numbers[left]+numbers[right] == target:
                return (left+1, right+1)
```

--------------------------------------------------------

## 169. Majority Element <a name="169"></a>

**Solution 1:** One solution is to use hash table. `O(n)` time complexity, and `O(n)` space complexity.

```python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dict = {}
        max_num = 0
        max_index = -1
        for i in nums:
        	if i in dict:
        		dict[i] = dict[i]+1
        	else:
        		dict[i] = 1
        	if dict[i] > max_num:
        		max_num = dict[i]
        		max_index = i
        return max_index
```

**Solution 2:** Another more efficient solution is to use Boyer-Moore Voting Algorithm, which can achieve `O(n)` time complexity and `O(1)` space complexity. However, this algorithm outputs wrong result when the majority does not exist.

```python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = 0
        candidate = None

        for num in nums:
        	if count == 0:
        		candidate = num
        	count += (1 if num==candidate else -1)
        return candidate
```

--------------------------------------------------------

## 242. Valid Anagram <a name="242"></a>

**Solution:** hash table

```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        dict = {}
        for i in s:
            if i not in dict:
                dict[i] = 1
            else:
                dict[i] += 1

        for i in t:
            if i not in dict:
                return False
            else:
                dict[i] -= 1

        for key, value in dict.items():
            if value!=0:
                return False
        return True
```

--------------------------------------------------------

## 278. First Bad Version <a name="278"></a>

**Solution:** binary search.

```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):
class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left = 1
        right = n
        while right > left:
        	mid = int((left+right)/2)
        	if isBadVersion(mid):
        		right = mid
        	else:
        		left = mid+1
        return left
```

--------------------------------------------------------

## 349. Intersection of Two Arrays

**Solution:** hash table

```python
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        dict = {}
        result = []
        for i in nums1:
            if i not in dict:
                dict[i] = 1

        for j in nums2:
            if j in dict and dict[j]==1:
                result.append(j)
                dict[j] -= 1
        return result
```

--------------------------------------------------------

## 350. Intersection of Two Arrays II <a name="350"></a>

**Solution:** hash table.

```python
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        dict = {}
        for i in nums1:
        	if i in dict:
        		dict[i] +=1
        	else:
        		dict[i] = 1

        result = []
        for i in nums2:
        	if i in dict and dict[i]>0:
        		result.append(i)
        		dict[i] -=1

        return result
```

--------------------------------------------------------

## 687. Longest Univalue Path <a name="687"></a>

**Solution:** This problem is categorized into an easy problem, but it requires careful design of the programming. The challenge is that the longest path can be either one direction or an arrow shape. In our design, we use a recursive function to return the longest one direction path, and use a **globel variable** to record the longest length of the arrow shape path. Doing so can greatly simply the recursive function, since it does not need to maintain two variables.

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    ans = 0 # it is very important to have a global variable

    def searchPath(self, root):
        left_len, right_len = 0, 0
        left_arr, right_arr = 0, 0
        if root.left != None:
            left_len = self.searchPath(root.left)
            if root.val == root.left.val:
                left_arr = left_len + 1
        if root.right != None:
            right_len = self.searchPath(root.right)
            if root.val == root.right.val:
                right_arr = right_len + 1
        self.ans = max(self.ans, left_arr + right_arr)
        return max(left_arr, right_arr)

    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        self.searchPath(root)
        return self.ans
```

--------------------------------------------------------
