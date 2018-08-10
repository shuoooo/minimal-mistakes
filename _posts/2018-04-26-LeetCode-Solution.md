---
title: "My LeetCode Solutions"
mathjax: true
---

This document tracks the LeetCode problems that I have finished, as well as my solutions. I use python 3.4 in the following solutions, though the versions submitted to LeetCode platform could be slightly modified.

## Table of contents
[1. Two Sum](#1) <br/>
[2. Add Two Numbers](#2) <br/>
[3. Longest Substring Without Repeating Characters](#3) <br/>
[5. Longest Palindromic Substring](#5) <br/>
[6. ZigZag Conversion](#6) <br/>
[7. Reverse Integer](#7) <br/>
[8. String to Integer (atoi)](#8) <br/>
[9. Palindrome Number](#9) <br/>
[12. Integer to Roman](#12) <br/>
[13. Roman to Integer](#13) <br/>
[14. Longest Common Prefix](#14) <br/>
[15. 3Sum](#15) <br/>
[16. 3Sum Closest](#16) <br/>
[17. Letter Combinations of a Phone Number](#17) <br/>
[18. 4Sum](#18) <br/>
[19. Remove Nth Node From End of List](#19) <br/>
[20. Valid Parentheses](#20) <br/>
[21. Merge Two Sorted Lists](#21) <br/>
[22. Generate Parentheses](#22) <br/>
[26. Remove Duplicates from Sorted Array](#26) <br/>
[27. Remove Element](#27) <br/>
[28. Implement strStr()](#28) <br/>
[29. Divide Two Integers](#29) <br/>
[30. Next Permutation](#31) <br/>
[32. Longest Valid Parentheses](#32) <br/>
[33. Search in Rotated Sorted Array](#33) <br/>
[34. Search for a Range](#34) <br/>
[35. Search Insert Position](#35) <br/>
[36. Valid Sudoku](#36) <br/>
[38. Count and Say](#38) <br/>
[39. Combination Sum](#39) <br/>
[40. Combination Sum II](#40) <br/>
[43. Multiply Strings](#43) <br/>
[46. Permutations](#46) <br/>
[47. Permutations II](#47) <br/>
[48. Rotate Image](#48) <br/>
[49. Group Anagrams](#49) <br/>
[50. Pow(x,n)](#50) <br/>
[53. Maximum Subarray](#53) <br/>
[54. Spiral Matrix](#54) <br/>
[55. Jump Game](#55) <br/>
[56. Merge Intervals](#56) <br/>
[58. Length of Last Word](#58) <br/>
[59. Spiral Matrix II](#59) <br/>
[60. Permutation Sequence](#60) <br/>
[61. Rotate List](#61) <br/>
[62. Unique Paths](#62) <br/>
[63. Unique Paths II](#63) <br/>
[64. Minimum Path Sum](#64) <br/>
[66. Plus One](#66) <br/>
[67. Add Binary](#67) <br/>
[69. Sqrt(x)](#69) <br/>
[70. Climbing Stairs](#70) <br/>
[71. Simplify Path](#71) <br/>
[72. Edit Distance](#72) <br/>
[73. Set Matrix Zeroes](#73) <br/>
[74. Search a 2D Matrix](#74) <br/>
[75. Sort Colors](#75) <br/>
[77. Combinations](#77) <br/>
[83. Remove Duplicates from Sorted List](#83) <br/>
[87. Scramble String](#87) <br/>
[88. Merge Sorted Array](#88) <br/>
[89. Gray Code](#89) <br/>
[90. Subsets II](#90) <br/>
[91. Decode Ways](#91) <br/>
[100. Same Tree](#100) <br/>
[101. Symmetric Tree](#101) <br/>
[104. Maximum Depth of Binary Tree](#104) <br/>
[107. Binary Tree Level Order Traversal II](#107) <br/>
[108. Convert Sorted Array to Binary Search Tree](#108) <br/>
[110. Balanced Binary Tree](#110) <br/>
[111. Minimum Depth of Binary Tree](#111) <br/>
[112. Path Sum](#112) <br/>
[118. Pascal\'s Triangle](#118) <br/>
[119. Pascal\'s Triangle II](#119) <br/>
[121. Best Time to Buy and Sell Stock](#121) <br/>
[122. Best Time to Buy and Sell Stock II](#122) <br/>
[123. Best Time to Buy and Sell Stock III](#123) <br/>
[125. Valid Palindrome](#125) <br/>
[129. Sum Root to Leaf Numbers](#129) <br/>
[130. Surrounded Regions](#130) <br/>
[136. Single Number](#136) <br/>
[137. Single Number II](#137) <br/>
[152. Maximum Product Subarray](#152) <br/>
[167. Two Sum II - Input array is sorted](#167) <br/>
[168. Excel Sheet Column Title](#168) <br/>
[169. Majority Element](#169) <br/>
[172. Factorial Trailing Zeroes](#172) <br/>
[198. House Robber](#198) <br/>
[213. House Robber II](#213) <br/>
[217. Contains Duplicate](#217) <br/>
[219. Contains Duplicate II](#219) <br/>
[242. Valid Anagram](#242) <br/>
[278. First Bad Version](#278) <br/>
[349. Intersection of Two Arrays](#349) <br/>
[350. Intersection of Two Arrays II](#350) <br/>
[415. Add Strings](#415) <br/>
[687. Longest Univalue Path](#687) <br/>
[698. Partition to K Equal Sum Subsets](#698) <br/>

--------------------------------------------------------

### 1. Two Sum <a name="1"></a>

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

### 2. Add Two Numbers <a name="2"></a>

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

### 3. Longest Substring Without Repeating Characters <a name="3"></a>

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

### 5. Longest Palindromic Substring <a name="5"></a>
**Solution:** use dynamic programming, $$O(n^2)$$ time and space complexity.
```python
class Solution(object):
	def longestPalindrome(self, s):
		"""
		:type s: str
		:rtype: str
		"""
		if len(s)<2:
			return s

		f = [[0]*len(s) for i in range(len(s))]
		for j in range(len(s)):
			f[0][j] = 1
		max_len, max_str = 1, s[0]

		for j in range(len(s)-1):
			f[1][j] = int(s[j] == s[j+1])
			if 2 > max_len and f[1][j] ==1:
				max_len, max_str = 2, s[j:j+2]

		for i in range(2,len(s)): # length is i+1
			for j in range(len(s)-i):
				if s[j]==s[j+i] and f[i-2][j+1]==1:
					f[i][j] = 1
					if i+1 > max_len:
						max_str = s[j:j+i+1]
		return max_str
```

--------------------------------------------------------

### 6. ZigZag Conversion <a name="6"></a>
```python
class Solution(object):
	def convert(self, s, numRows):
		if numRows == 1:
			return s
		result = [[] for i in range(numRows)]
		denominator = numRows*2-2
		for i in range(len(s)):
			mod = i % denominator
			if mod < numRows:
				result[mod].append(s[i])
			else:
				result[denominator-mod].append(s[i])
		#print(result)

		output = ""
		for i in result:
			output = output + "".join(i)
		return output
```

--------------------------------------------------------

### 7. Reverse Integer <a name="7"></a>

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

### 8. String to Integer (atoi) <a name="8"></a>
```python
class Solution(object):
	def myAtoi(self, str):
		if len(str) == 0:
			return 0

		start = 0
		while(start < len(str) and str[start] == ' '):
			start += 1

		positive = 1
		if start < len(str) and str[start] in '+-':
			if str[start] == '-':
				positive = -1
			start += 1

		sum = 0
		upper = 2**31-1
		lower = -2**31
		for i in range(start, len(str)):
			if str[i] == ' ' or ord(str[i]) < ord('0') or ord(str[i]) > ord('9'):
				break
			sum = sum*10  + ord(str[i]) - ord('0')
			if positive == 1 and sum > upper:
				return upper
			elif positive == -1 and -sum < lower:
				return lower
		return sum*positive
```

--------------------------------------------------------

### 9. Palindrome Number <a name="9"></a>

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

### 12. Integer to Roman <a name="12"></a>
```python
class Solution(object):
	def converter(self, num, a, b, c):
		if num == 0:
			return ""
		if num<4:
			return a*num
		if num == 4:
			return a+b
		if num == 5:
			return b
		if num<9:
			return b+a*(num-5)
		else:
			return a+c

	def intToRoman(self, num):
		#d = {1:'I', 5:'V', 10:'X', 50:'L', 100:'C', 500:'D', 1000:'M'}
		thous = num//1000
		hund = (num%1000)//100
		ten = (num%100)//10
		one = (num%10)

		output = "" + 'M'*thous
		output += self.converter(hund, 'C', 'D', 'M')
		output += self.converter(ten, 'X', 'L', 'C')
		output += self.converter(one, 'I', 'V', 'X')
		return output
```

--------------------------------------------------------

### 13. Roman to Integer <a name="13"></a>

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

### 14. Longest Common Prefix <a name="14"></a>

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

### 15. 3Sum <a name="15"></a>

**Solution**: when one number is fixed, the problem becomes two-sum problem. The only difference is that we need to pay attention to the duplicates.

```python
class Solution(object):
	def threeSum(self, nums):
		result = []
		nums.sort()
		for i in range(len(nums)-2):
			if i > 0 and nums[i] == nums[i-1]:
				continue
			left, right = i+1, len(nums)-1
			while(left < right):
				sum = nums[i] + nums[left] + nums[right]
				if sum < 0:
					left += 1
				elif sum > 0:
					right -= 1
				else:
					left_num = nums[left]
					right_num = nums[right]
					result.append([nums[i], left_num, right_num])
					while(left < right and nums[left]==left_num):
						left += 1
					while(left < right and nums[right]==right_num):
						right -= 1
		return result
```

--------------------------------------------------------

### 16. 3Sum Closest <a name="16"></a>
**Solution** Similar to 3sum
```python
class Solution(object):
	def threeSumClosest(self, nums, target):
		result = target + 99999
		nums.sort()

		for i in range(len(nums)-2):
			if i > 0 and nums[i] == nums[i-1]:
				continue
			left, right = i+1, len(nums)-1
			while(left < right):
				sum = nums[i] + nums[left] + nums[right]
				result = result if abs(result-target) < abs(sum-target) else sum
				if sum < target:
					left += 1
				elif sum > target:
					right -= 1
				else:
					return target
		return result
```
--------------------------------------------------------
### 17. Letter Combinations of a Phone Number <a name="17"></a>
**Solution:** Recursive with DFS
```python
class Solution(object):
	d = {"2":'abc', "3":'def', "4":'ghi', "5":'jkl', "6":'mno',"7":'pqrs', "8":'tuv', "9":'wxyz'}
	result = []
	def letterCombinationsSub(self, digits, seq):
		if len(digits)==0:
			self.result.append(seq)
			return
		for i in self.d[digits[0]]:
			self.letterCombinationsSub(digits[1:], seq+i)

	def letterCombinations(self, digits):
		if len(digits)==0:
			return []
		self.result = []
		self.letterCombinationsSub(digits,"")
		return self.result
```
--------------------------------------------------------

### 18. 4Sum <a name="18"></a>
**Solution:** Similar to 3Sum, only add another outer loop.
```python
class Solution(object):
	def fourSum(self, nums, target):
		if len(nums) < 4:
			return []

		nums.sort()
		result = []
		for i in range(len(nums)-3):
			if i > 0 and nums[i]==nums[i-1]:
				continue
			for j in range(i+1, len(nums)-2):
				if j > i+1 and nums[j]==nums[j-1]:
					continue

				left = j + 1
				right = len(nums) - 1
				while(left < right):
					left_num = nums[left]
					right_num = nums[right]
					sum = nums[i] + nums[j] + left_num + right_num
					if sum < target:
						left += 1
					elif sum > target:
						right -= 1
					else:
						temp = [nums[i], nums[j], left_num, right_num]
						result.append(temp)
						while(left < right and nums[left]==left_num):
							left += 1
						while(left < right and nums[right]==right_num):
							right -= 1
		return result
```
-------------------------------------------------------

### 19. Remove Nth Node From End of List <a name="19"></a>
**Solution:** easy, use stack
```python
class Solution(object):
	def removeNthFromEnd(self, head, n):
		if n==0:
			return head
		if head.next == None and n == 1:
			return None
		stack = []
		p = head
		while(p.next != None):
			stack.append(p)
			p = p.next

		if n==1:
			stack[-1].next = None
		elif n <= len(stack):
			stack[-n].next = stack[-n].next.next
		else:
			head = head.next
		return head
```

-----------------------------------------------------

### 20. Valid Parentheses <a name="20"></a>

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

### 21. Merge Two Sorted Lists <a name="21"></a>

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

### 22. Generate Parentheses <a name="22"></a>
```python
class Solution(object):
	result = []
	def generateParenthesisSub(self, left, right, seq):
		for i in range(1,left):
			for j in range(1,i+right-left+1):
				self.generateParenthesisSub(left-i, right-j, seq + '('*i + ')'*j)
		self.result.append(seq+'('*left+')'*right)

	def generateParenthesis(self, n):
		if n==0:
			return []
		self.result = []
		self.generateParenthesisSub(n,n,"")
		return self.result
```

--------------------------------------------------------
### 24. Swap Nodes in Pairs <a name="24"></a>
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
	def swapPairs(self, head):
		if head == None or head.next == None:
			return head
		a, b = head, head.next
		head, p = b, None
		while(1):
			a.next = b.next
			b.next = a
			if p != None:
				p.next = b
			p = a

			if a.next == None or a.next.next == None:
				break
			else:
				a = a.next
				b = a.next
		return head
```
--------------------------------------------------------

### 26. Remove Duplicates from Sorted array <a name="26"></a>

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

### 27. Remove Element <a name="27"></a>

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

### 28. Implement strStr() <a name="28"></a>

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
### 29. Divide Two Integers <a name="29"></a>
**Solution:** bit operation
```python
class Solution(object):
	def divide(self, dividend, divisor):
		positive = (dividend<0) == (divisor<0)
		dividend, divisor = abs(dividend), abs(divisor)
		res = 0
		while(dividend >= divisor):
			temp = divisor
			i = 1
			while(dividend >= temp):
				dividend -= temp
				res += i
				temp <<= 1
				i <<= 1
		if not positive:
			res = -res
		return min(max(-2**31,res),2**31-1)
```
--------------------------------------------------------

### 31. Next Permutation
**Solution:** classic problem. See `nextPermutation` and `prevPermutation` below.
```python
class Solution(object):
	def nextPermutation(self, nums):
		'''
		1. from end to start, find the first two adjacent (i,j) such that nums[i] < nums[j]
		2. in nums[j:end], find the smallest nums[k] such that nums[k] > nums[i]
		3. swap nums[i] and nums[k]
		4. reverse nums[j:end]
		'''
		i = len(nums)-2
		while i>=0 and nums[i] >= nums[i+1]:
			i -= 1
		if i < 0:
			nums.sort()
			return False

		k = len(nums)-1
		while(nums[k] <= nums[i]):
			k -= 1

		nums[i], nums[k] = nums[k], nums[i]
		nums[i+1:] = nums[i+1:][::-1]
		return True

	def prevPermutation(self, nums):
		'''
		previous permutation is actually the next permutation of the inverselly sorted nums,
		thus we only need to change the orders in nextPermutation function.
		'''
		i = len(nums)-2
		while i>=0 and nums[i] <= nums[i+1]:
			i -= 1
		if i < 0:
			nums.sort(reverse=True)
			return False

		k = len(nums)-1
		while(nums[k] >= nums[i]):
			k -= 1

		nums[i], nums[k] = nums[k], nums[i]
		nums[i+1:] = nums[i+1:][::-1]
		return True
```

--------------------------------------------------------

### 32. Longest Valid Parentheses <a name="32"></a>

**Solution1:** using stack.
```python
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_len = 0
        stack = [-1]
        for i in range(len(s)):
        	if s[i] == '(':
        		stack.append(i)
        	else:
        		stack.pop()
        		if len(stack) == 0:
        			stack.append(i)
        		else:
        			max_len = max(max_len, i - stack[-1])
        return max_len

```

**Solution2:** left-right scan.
```python
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, right, max_len = 0, 0, 0
        for i in range(len(s)):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left==right:
                max_len = max(max_len, left*2)
                prev = left
            elif right > left:
                left, right = 0, 0
        left, right = 0, 0
        for i in reversed(range(len(s))):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left==right:
                max_len = max(max_len, left*2)
                prev = left
            elif left > right:
                left, right = 0, 0
        return max_len
```
--------------------------------------------------------

### 33. Search in Rotated Sorted Array <a name="33"></a>
**Solution:** first find the pivot point with binary search, then binary search for the target point
```python
class Solution(object):
	def findPivotPoine(self, nums):
		low, hi = 0, len(nums)-1
		while(low<hi):
			mid = (low+hi)//2
			if mid+1 < len(nums) and nums[mid] > nums[mid+1]:
				return mid
			if nums[mid] < nums[-1]:
				hi = mid
			else:
				low = mid
		return len(nums)-1

	def search(self, nums, target):
		if len(nums) == 0:
			return -1
		pivotIndex = self.findPivotPoine(nums)

		if pivotIndex == len(nums)-1:
			low, hi = 0, len(nums)-1
		elif  target > nums[-1]:
			low, hi = 0, pivotIndex
		else:
			low, hi = pivotIndex+1, len(nums)-1

		while(low < hi):
			mid = (low+hi)//2
			if nums[mid] == target:
				return mid
			if nums[mid] > target:
				hi = mid
			else:
				low = mid+1

		if low == hi and target == nums[low]:
			return low
		return -1
```
--------------------------------------------------------

### 34. Search for a Range <a name="34"></a>

**Solution:** first binary search for left boundary, then right boundary
```python
class Solution(object):
	def searchRange(self, nums, target):
		if not nums:
			return [-1,-1]
		left, right = 0, len(nums)-1
		while(left<right):
			mid = (left+right)//2
			if nums[mid] < target:
				left = mid+1
			else:
				right = mid

		if nums[left] == target:
			left_index = left
		else:
			return [-1,-1]

		left, right = left, len(nums)-1
		while(left<right):
			mid = (left+right)//2
			if nums[mid] > target:
				right = mid
			else:
				left = mid+1

		if nums[right] == target:
			return [left_index, right]
		else:
			return [left_index, right-1]
```

--------------------------------------------------------

### 35. Search Insert Position <a name="35"></a>

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
### 36. Valid Sudoku <a name="36"></a>
**Solution:** this problem is easy
```python
class Solution(object):
	def checkEntity(self, entity):
		d = {}
		for i in entity:
			if ord(i) >= ord("0") and ord(i) <= ord("9"):
				if i not in d:
					d[i] = 1
				else:
					return False
		return True

	def isValidSudoku(self, board):
		for row in board:
			if not self.checkEntity(row):
				return False
		for i in range(9):
			col = [row[i] for row in board]
			if not self.checkEntity(col):
				return False
		for i in range(3):
			for j in range(3):
				subbox = board[3*i][3*j:3*j+3] + board[3*i+1][3*j:3*j+3] + board[3*i+2][3*j:3*j+3]
				if not self.checkEntity(subbox):
					return False
		return True
```
--------------------------------------------------------

### 38. Count and Say <a name="38"></a>

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
### 39. Combination Sum <a name="39"></a>
```python
class Solution(object):
	ans = []
	def subCS(self, candidates, target, prev):
		if target == 0 and len(prev) != 0:
			self.ans.append(prev)
		for i in reversed(range(len(candidates))):
			if candidates[i] > target:
				continue
			self.subCS(candidates[:i+1], target-candidates[i], prev+[candidates[i]])

	def combinationSum(self, candidates, target):
		self.ans = []
		candidates.sort()
		self.subCS(candidates, target, [])
		return self.ans
```
--------------------------------------------------------
### 40. Combination Sum II <a name="40"></a>
```python
class Solution(object):
	ans = []
	def subCS(self, candidates, target, prev):
		if target == 0 and len(prev) != 0 and prev not in self.ans:
			self.ans.append(prev)
		for i in reversed(range(len(candidates))):
			if candidates[i] > target:
				continue
			self.subCS(candidates[:i], target-candidates[i], prev+[candidates[i]])

	def combinationSum2(self, candidates, target):
		self.ans = []
		candidates.sort()
		self.subCS(candidates, target, [])
		return self.ans
```
--------------------------------------------------------
### 43. Multiply Strings <a name='43'></a>
**Solution:** `n1[i] * n2[j]` will only affect `ans[i+j,i+j+1]`
```python
class Solution(object):
	def multiply(self, num1, num2):
		if num1 == '0' or num2 == '0':
			return '0'
		ans = [0] * (len(num1)+len(num2))
		for i in range(1,len(num1)+1):
			for j in range(1,len(num2)+1):
				mul = (ord(num1[-i]) - ord('0')) * (ord(num2[-j]) - ord('0'))
				sum = mul + ans[-i-j+1]
				ans[-i-j+1] = sum % 10
				ans[-i-j] += sum //10
		ans = [str(i) for i in ans]
		ans = "".join(ans)

		if ans[0] != '0':
			return ans
		else:
			return ans[1:]
```
--------------------------------------------------------
### 46. Permutations <a name="46"></a>
```python
class Solution(object):
	def addPerm(self, nums, prev, ans):
		if len(nums) == 0:
			ans.append(prev)
		for i in range(len(nums)):
			self.addPerm(nums[:i]+nums[i+1:], prev + [nums[i]], ans)

	def permute(self, nums):
		ans = []
		if len(nums) != 0:
			self.addPerm(nums, [], ans)
		return ans
```
--------------------------------------------------------
### 47. Permutations II <a name='47'></a>
```python
class Solution(object):
	def bt(self, nums, prev, ans):
		if len(nums) == 0:
			ans.append(prev)
		for i in range(len(nums)):
			if i > 0 and nums[i] == nums[i-1]:
				continue
			self.bt(nums[:i]+nums[i+1:], prev+[nums[i]], ans)

	def permuteUnique(self, nums):
		ans = []
		if len(nums) != 0:
			nums.sort()
			self.bt(nums, [], ans)
		return ans
```
--------------------------------------------------------
### 48. Rotate Image <a name="48"></a>
**Solution:** the code below can be simplified.
```python
class Solution(object):
	def position(self, matrix, cir, num, l):
		x0, y0 = cir, cir
		if num < l:
			return x0, y0+num
		elif num < 2*l:
			return x0+num-l, y0+l
		elif num < 3*l:
			return x0+l, y0+3*l-num
		else:
			return x0+4*l-num, y0

	def rotate(self, matrix):
		cir_num = (len(matrix)+1)//2
		for cir in range(cir_num):
			l = len(matrix) -2*cir -1
			for j in range(l):
				x0,y0 = self.position(matrix, cir, j, l)
				x1,y1 = self.position(matrix, cir, j+l, l)
				x2,y2 = self.position(matrix, cir, j+2*l, l)
				x3,y3 = self.position(matrix, cir, j+3*l, l)
				matrix[x0][y0], matrix[x1][y1], matrix[x2][y2], matrix[x3][y3] =\
				matrix[x3][y3], matrix[x0][y0], matrix[x1][y1], matrix[x2][y2]
```
--------------------------------------------------------
### 49. Group Anagrams <a name="49"></a>
**Solution 1:** sort each strings
```python
class Solution(object):
	def groupAnagrams(self, strs):
		d = {}
		for i in strs:
			s = "".join(sorted(i))
			if s not in d:
				d[s] = [i]
			else:
				d[s].append(i)
		return list(d.values())
```
**Solution 2:** prime coding
```python
class Solution(object):
	def groupAnagrams(self, strs):
		prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
		d = {}
		for i in strs:
			num = 1
			for j in i:
				num *= prime[ord(j) - ord('a')]

			if num not in d:
				d[num] = [i]
			else:
				d[num].append(i)
		return list(d.values())
```
--------------------------------------------------------
### 50. Pow(x,n) <a name="50"></a>
```python
class Solution(object):
	def myPow(self, x, n):
		if n==0:
			return 1
		if n<0:
			n = -n
			x = 1.0/x

		bits = "{0:b}".format(n)
		temp, ans = x, 1
		for i in range(len(bits)):
			if bits[~i] == '1':
				ans *= temp
			temp *= temp
		return ans
```

--------------------------------------------------------
### 53. Maximum Subarray <a name="53"></a>
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
### 54. Spiral Matrix <a name="54"></a>
```python
class Solution(object):
	def spiralSingle(self, matrix, ans):
		if len(matrix) != 0:
			ans += matrix[0]
			if len(matrix) > 1:
				ans += [matrix[i][-1] for i in range(1,len(matrix)-1)]
				ans += matrix[-1][::-1]
			if len(matrix[0]) > 1:
				ans += [matrix[i][0] for i in reversed(range(1,len(matrix)-1))]
			submatrix = [row[1:-1] for row in matrix[1:len(matrix)-1] if len(row)>2]
			self.spiralSingle(submatrix, ans)

	def spiralOrder(self, matrix):
		ans = []
		self.spiralSingle(matrix, ans)
		return ans
```
--------------------------------------------------------
### 55. Jump Game <a name="55"></a>
```python
class Solution(object):
	def canJump(self, nums):
		right, l, i = nums[0], len(nums), 0
		while i<right+1:
			if i+nums[i] > right:
				right = i+nums[i]
			if right >= l-1:
				return True
			i += 1
		return right>=l-1
```
--------------------------------------------------------
### 56. Merge Intervals <a name="56"></a>
```python
class Solution(object):
	def merge(self, intervals):
		intervals = sorted(intervals, key = lambda x: x.start)
		ans = []
		for x in intervals:
			if len(ans)>0 and ans[-1].end >= x.start:
				ans[-1].end = max(ans[-1].end, x.end)
			else:
				ans.append(Interval(x.start, x.end))
		return ans
```
--------------------------------------------------------
### 58. Length of Last Word <a name="58"></a>

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
-------------------------------------------------------
### 59. Spiral Matrix II <a name="59"></a>
```python
class Solution(object):
	def generateMatrix(self, n):
		ans = [[0]*n for i in range(n)]
		cirNum = (n+1)//2
		num = 1
		for c in range(cirNum):
			l = n - c*2 -1
			x, y = c, c

			if l == 0:
				ans[x][y] = num
			for i in range(l):
				ans[x][y] = num
				y += 1
				num += 1
			for i in range(l):
				ans[x][y] = num
				x += 1
				num += 1
			for i in range(l):
				ans[x][y] = num
				y -= 1
				num += 1
			for i in range(l):
				ans[x][y] = num
				x -= 1
				num += 1
		return ans
```
--------------------------------------------------------
### 60. Permutation Sequence <a name="60"></a>
**Solution:** we observe that, if $$k <= (n-1)!$$, then the kth permutation must start with 1, and if $$(n-1)! < k <= 2*(n-1)!$$, the kth permutation must starts with 2.
```python
class Solution(object):
	def getPermutation(self, n, k):
		factorial = [0,1]
		for i in range(2,n+1):
			factorial.append(factorial[-1] * i)

		if n==0 or k==0 or k> factorial[n]:
			return []

		num = [str(i) for i in range(1,n+1)]
		ans = []

		for i in reversed(range(2,n+1)):
			if k == factorial[i]:
				num.sort(reverse=True)
				break
			if k == 0:
				break

			divisor = (k-1)//factorial[i-1]
			ans.append(num[divisor])
			num.pop(divisor)
			k -= divisor*factorial[i-1]

		ans = ans + num
		return "".join(ans)
```
--------------------------------------------------------
### 61. Rotate List <a name="61"></a>
```python
class Solution(object):
	def rotateRight(self, head, k):
		if not head:
			return head
		length = 1
		p = head
		while(p.next!=None):
			length += 1
			p = p.next
		tail = p

		k = k % length
		ans = head
		if k != 0:
			position = length - k + 1
			p = head
			i = 1
			while(i < position-1):
				p = p.next
				i += 1
			ans = p.next
			p.next = tail.next
			tail.next = head
		return ans
```
--------------------------------------------------------
### 62. Unique Paths <a name="62"></a>
```python
class Solution(object):
	def uniquePaths(self, m, n):
		if n > m:
			m, n = n, m
		m, n = m-1, n-1

		result = 1
		for i in range(n):
			result = result * (m+n-i) / (i+1)
		return int(result)
```
--------------------------------------------------------
### 63. Unique Paths II <a name="63"></a>
```python
class Solution(object):
	def uniquePathsWithObstacles(self, obstacledGrid):
		m, n = len(obstacledGrid), len(obstacledGrid[0])
		ans = [[0]*n for i in range(m)]
		if obstacledGrid[0][0] == 1:
			return 0
		else:
			ans[0][0] = 1

		for i in range(m):
			for j in range(n):
				if obstacledGrid[i][j] == 0:
					if i > 0:
						ans[i][j] += ans[i-1][j]
					if j > 0:
						ans[i][j] += ans[i][j-1]
		return ans[-1][-1]
```
--------------------------------------------------------
### 64. Minimum Path Sum <a name="64"></a>
```python
class Solution(object):
	def minPathSum(self, grid):
		m, n = len(grid), len(grid[0])
		for i in range(m):
			for j in range(n):
				if i == 0 and j != 0:
					grid[i][j] += grid[i][j-1]
				elif i != 0 and j == 0:
					grid[i][j] += grid[i-1][j]
				elif i != 0 and j != 0:
					grid[i][j] += min(grid[i-1][j], grid[i][j-1])
		return grid[-1][-1]

```
--------------------------------------------------------
### 66. Plus One <a name="66"></a>

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
### 67. Add Binary <a name="67"></a>

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

### 69. Sqrt(x) <a name="69"></a>

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

### 70. Climbing Stairs <a name="70"></a>

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
### 71. Simplify Path <a name="71"></a>
```python
class Solution(object):
	def simplifyPath(self, path):
		x = path.split('/')
		ans = []
		for i in x:
			if i == '' or i == '.':
				continue
			if i == '..':
				if len(ans) > 0:
					ans.pop()
			else:
				ans.append(i)
		ans = "/".join(ans)
		return "/"+ans
```
--------------------------------------------------------

### 72. Edit Distance <a name="72"></a>

**Solution1:** DP with $$O(mn)$$ space complexity
```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """

        l1, l2 = len(word1), len(word2)
        f = [[0]*(l2+1) for i in range(l1+1)]
        for i in range(1,l1+1):
        	f[i][0] = i
        for j in range(1,l2+1):
        	f[0][j] = j
        for i in range(1,l1+1):
        	for j in range(1,l2+1):
        		f[i][j] = min(min(f[i-1][j]+1, f[i][j-1]+1), f[i-1][j-1]+(word1[i-1]!=word2[j-1]))
        return f[l1][l2]
```

**Solution2:** DP with $$O(m)$$ space complexity.
```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """

        l1, l2 = len(word1), len(word2)
        f = list(range(l2+1))

        for i in range(1,l1+1):
            prev = f[0]
            f[0] = i
            for j in range(1,l2+1):
                temp = f[j]
                f[j] = min(min(f[j]+1, f[j-1]+1), prev+(word1[i-1]!=word2[j-1]))
                prev = temp
        return f[-1]
```

--------------------------------------------------------
### 73. Set Matrix Zeroes <a name="73"></a>
**Solution:** constant space solution. Store the state of each row/col in the first place of the row/col. The exception is [0,0] is the intersection of the first row and first col, so we store the states of first row/col in two variabels: row0, col0.
```python
class Solution(object):
	def setZeroes(self, matrix):
		m, n = len(matrix), len(matrix[0])
		row0, col0 = False, False
		for i in range(m):
			for j in range(n):
				if i==0 and matrix[i][j]==0:
					row0 = True
				if j==0 and matrix[i][j]==0:
					col0 = True
				if matrix[i][j] == 0:
					matrix[i][0] = 0
					matrix[0][j] = 0

		for i in range(1,m):
			if matrix[i][0] == 0:
				matrix[i] = [0]*n
		for j in range(1,n):
			if matrix[0][j] == 0:
				for row in matrix:
					row[j] = 0
		if row0:
			matrix[0] = [0]*n
		if col0:
			for row in matrix:
				row[0] = 0
```
--------------------------------------------------------
### 74. Search a 2D Matrix <a name="74"></a>
**Solution:** treat it like a sorted list, the complexity is $$O(log(mn)) = O(log(m)+log(n))$$, which is exactly the same as first searching for row, then searching for col.
```python
class Solution(object):
	def searchMatrix(self, matrix, target):
		if len(matrix)==0 or len(matrix[0])==0 or target<matrix[0][0] or target > matrix[-1][-1]:
			return False
		m, n = len(matrix), len(matrix[0])
		left, right = 0, m*n-1

		while(left < right):
			mid = (left+right)//2
			if matrix[mid//n][mid%n] == target:
				return True
			elif matrix[mid//n][mid%n] < target:
				left = mid + 1
			else:
				right = mid - 1
		return matrix[left//n][left%n] == target
```
--------------------------------------------------------
### 75. Sort Colors <a name="75"></a>
**Solution1:** A rather straight forward solution is a two-pass algorithm using counting sort. First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
```python
class Solution(object):
	def sortColors(self, nums):
		count = [0, 0, 0]
		for i in nums:
			count[i] += 1
		for i in range(len(nums)):
			if i < count[0]:
				nums[i] = 0
			elif i < count[0]+count[1]:
				nums[i] = 1
			else:
				nums[i] = 2
```
**Solution2:** One phase
```python
class Solution(object):
	def sortColors(self, nums):
		zero, one = 0, 0
		for i in range(len(nums)):
			x = nums[i]
			nums[i] = 2
			if x < 2:
				nums[one] = 1
				one += 1
			if x == 0:
				nums[zero] = 0
				zero += 1
```
--------------------------------------------------------
### 77. Combinations <a name="77"></a>
```python
class Solution(object):
	def getCombine(self, n, k, index, prev, ans):
		if len(prev)==k:
			ans.append(prev)
			return
		for i in range(index,n-(k-len(prev))+2):
			self.getCombine(n, k, i+1, prev+[i], ans)

	def combine(self, n, k):
		ans = []
		if k <= n:
			self.getCombine(n, k, 1, [], ans)
		return ans
```
--------------------------------------------------------
### 78. Subsets <a name="78"></a>
```python
class Solution(object):
	def subsets(self, nums):
		ans = []
		size = 2**len(nums)
		for i in range(size):
			x = "{0:b}".format(i)
			result = []
			for j in range(len(x)):
				if x[j] == '1':
					result.append(nums[len(nums)-len(x)+j])
			ans.append(result)
		return ans
```
--------------------------------------------------------
### 79. Word Search <a name="79"></a>
```python
class Solution(object):
	m = 0
	n = 0
	def adjacent(self, i, j):
		ans = []
		if i > 0: ans.append((i-1,j))
		if i < self.m-1: ans.append((i+1,j))
		if j > 0: ans.append((i,j-1))
		if j < self.n-1: ans.append((i,j+1))
		return ans

	def wordSearch(self, board, word, used, i, j):
		if len(word) == 0:
			return True
		adj = self.adjacent(i, j)
		for x, y in adj:
			if board[x][y] == word[0] and used[x][y] == 0:
				used[x][y] = 1
				if self.wordSearch(board, word[1:], used, x, y):
					return True
				used[x][y] = 0
		return False

	def exist(self, board, word):
		if len(word) == 0:
			return True
		if len(board) == 0:
			return False
		self.m, self.n = len(board), len(board[0])
		used = [[0]*self.n for i in range(self.m)]
		for i in range(self.m):
			for j in range(self.n):
				if board[i][j] == word[0]:
					used[i][j] = 1
					if self.wordSearch(board, word[1:], used, i, j):
						return True
					used[i][j] = 0
		return False
```
--------------------------------------------------------
### 80. Remove Duplicates from Sorted Array II <a name="80"></a>
```python
class Solution(object):
	def removeDuplicates(self, nums):
		if not nums:
			return 0
		p = 1
		flag = False
		for i in range(1,len(nums)):
			if nums[i] == nums[i-1]:
				if not flag:
					flag = True
					nums[p] = nums[i]
					p += 1
			else:
				flag = False
				nums[p] = nums[i]
				p += 1
		return p
```
--------------------------------------------------------
### 83. Remove Duplicates from Sorted Lists <a name="83"></a>

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

### 87. Scramble String <a name="87"></a>

**Solution:** Recursion.
```python
class Solution(object):
	def isScramble(self, s1, s2):
		if len(s1)!=len(s2) or sorted(s1)!=sorted(s2): # may use dict
			return False
		if s1==s2 or len(s1)==1:
			return True

		for i in range(1,len(s1)):
			a1, a2 = s1[:i], s1[i:]
			b1, b2 = s2[:i], s2[i:]
			c1, c2 = s2[-i:], s2[:-i]
			if (self.isScramble(a1,b1) and self.isScramble(a2,b2)) \
			 or (self.isScramble(a1,c1) and self.isScramble(a2,c2)):
				return True
		return False
```

--------------------------------------------------------

### 88. Merge Sorted Array <a name="88"></a>

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
### 89. Gray Code <a name="89"></a>
**Solution 1:** Using the rule that the gray code for number $$i$$ is $$i ^ i>>1$$.
```python
class Solution(object):
	def grayCode(self, n):
		ans = []
		for i in range(2**n):
			ans.append(i^(i>>1))
		return ans
```
**Solution 2:** The graycode sequence can be generated iteratively. For example, when n=3, we can get the result based on n=2: 00,01,11,10 -> (000,001,011,010) (110,111,101,100). The middle two numbers only differ at their highest bit, while the rest numbers of part two are exactly symmetric of part one.
```python
class Solution(object):
	def grayCode(self, n):
		ans = [0]
		for i in range(n):
			for j in reversed(range(len(ans))):
				ans.append(ans[j] | 1<<i)
		return ans
```
--------------------------------------------------------
### 90. Subsets II <a name="90"></a>
```python
class Solution(object):
	d, ans, unique = None, None, None

	def getSubset(self, code, result, index):
		if len(code) == 0:
			self.ans.append(result)
		else:
			if code[0] == '1':
				number = self.unique[index]
				for i in range(1,self.d[number]+1):
					self.getSubset(code[1:], result + [number]*i, index+1)
			else:
				self.getSubset(code[1:], result, index+1)

	def subsetsWithDup(self, nums):
		self.d, self.ans, self.unique = {}, [], []
		for i in nums:
			if i not in self.d:
				self.d[i] = 1
				self.unique.append(i)
			else:
				self.d[i] += 1

		for i in range(2**len(self.unique)):
			code = "{0:b}".format(i)
			if len(code) < len(self.unique):
				code = '0'*(len(self.unique)-len(code)) + code
			result = []
			self.getSubset(code, result, 0)

		return self.ans
```
--------------------------------------------------------
### 91. Decode Ways <a name="91"></a>
```python
class Solution(object):
	def decode(self, d, s, index):
		if len(s) == 0:
			return 1
		if s[0] == '0':
			d[index] = 0
			return 0
		if len(s) == 1:
			return 1
		if int(s[:2]) > 26:
			if index not in d:
				d[index] = self.decode(d, s[1:], index+1)
		else:
			if index not in d:
				d[index] = self.decode(d, s[1:], index+1) + self.decode(d, s[2:], index+2)
		return d[index]

	def numDecodings(self, s):
		if len(s) == 0:
			return 0
		d = {}
		return self.decode(d, s, 0)
```
--------------------------------------------------------
### 93. Restore IP Addresses <a name="93"></a>
```python
class Solution(object):
	def addIp(self, ans, s, result):
		if len(s) == 0 and len(result)==4:
			ans.append(".".join(result))
		else:
			if len(s) >= 1:
				self.addIp(ans, s[1:], result + [s[0]])
			if len(s) >=2 and s[0] != '0':
				self.addIp(ans, s[2:], result + [s[:2]])
			if len(s) >= 3 and int(s[:3]) <= 255 and s[0] != '0':
				self.addIp(ans, s[3:], result + [s[:3]])

	def restoreIpAddresses(self, s):
		ans = []
		if len(s) != 0 and len(s) <= 12:
			self.addIp(ans, s, [])
		return ans
```
--------------------------------------------------------
### 100. Same Tree <a name="100"></a>
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

### 101. Symmetric Tree <a name="101"></a>

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

### 104. Maximum Depth of Binary Tree <a name="104"></a>

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
### 107. Binary Tree Level Order Traversal II <a name="107"></a>
```python
class Solution(object):
	ans = []
	def treeSearch(self, subroot, level):
		if subroot:
			if level > len(self.ans):
				self.ans.insert(0,[])
			self.ans[-level].append(subroot.val)
			self.treeSearch(subroot.left, level+1)
			self.treeSearch(subroot.right, level+1)

	def levelOrderBottom(self, root):
		self.ans = []
		self.treeSearch(root, 1)
		return self.ans
```
--------------------------------------------------------
### 108. Convert Sorted Array to Binary Search Tree <a name="108"></a>
```python
class Solution(object):
	def getTree(self, nums):
		if len(nums) == 0:
			return None
		index = len(nums)//2
		root = TreeNode(nums[index])
		root.left = self.getTree(nums[0:index])
		root.right = self.getTree(nums[index+1:])
		return root

	def sortedArrayToBST(self, nums):
		return self.getTree(nums)
```
--------------------------------------------------------
### 110. Balanced Bianry Tree <a name="110"></a>
```python
class Solution(object):
	def getDepth(self, root):
		if root == None:
			return (0, True)

		if root.left == None and root.right == None:
			return (1, True)
		root.ldepth, flag1 = self.getDepth(root.left)
		root.rdepth, flag2 = self.getDepth(root.right)
		if not flag1 or not flag2:
			return (0, False)

		if abs(root.ldepth - root.rdepth) > 1:
			return (0, False)
		else:
			return (max(root.ldepth+1, root.rdepth+1), True)

	def isBalanced(self, root):
		if root == None:
			return True
		temp, ans = self.getDepth(root)
		return ans
```
--------------------------------------------------------
### 111. Minimum Depth of Binary Tree <a name="111"></a>
```python
class Solution(object):
	min = -1
	def getDepth(self, root, level):
		if root == None:
			self.min = 0
			return
		if root.left == None and root.right == None:
			if self.min == -1:
				self.min = level
			else:
				self.min = min(self.min, level)
		if root.left != None:
			self.getDepth(root.left, level+1)
		if root.right != None:
			self.getDepth(root.right, level+1)

	def minDepth(self, root):
		self.min = -1
		self.getDepth(root,1)
		return self.min
```
--------------------------------------------------------
### 112. Path Sum <a name="112"></a>
```python
class Solution(object):
	def subPS(self, subroot, sum, prev):
		if subroot.left == None and subroot.right == None:
			return prev + subroot.val == sum
		if subroot.left != None:
			if self.subPS(subroot.left, sum, prev+subroot.val):
				return True
		if subroot.right != None:
			if self.subPS(subroot.right, sum, prev+subroot.val):
				return True
		return False

	def hasPathSum(self, root, sum):
		if root == None:
			return False
		return self.subPS(root, sum, 0)
```
-------------------------------------------------------
### 113. Path Sum <a name="113"></a>
```python
class Solution(object):
	def getPath(self, ans, root, sum, path, prev):
		if root.left == None and root.right == None and prev+root.val == sum:
			ans.append(path+[root.val])
		if root.left != None:
			self.getPath(ans, root.left, sum, path+[root.val], prev+root.val)
		if root.right != None:
			self.getPath(ans, root.right, sum, path+[root.val], prev+root.val)

	def pathSum(self, root, sum):
		ans = []
		if root != None:
			self.getPath(ans, root, sum, [], 0)
		return ans
```
--------------------------------------------------------

### 118. Pascal\'s Triangle <a name="118"></a>

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

### 119. Pascal\'s Triangle II <a name="119"></a>

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
### 120. Triangle <a name="120"></a>
**Solution:** straightforward DP, using $$O(n^2)$$ space complexity, where $$n$$ is the number of rows.
```python
class Solution(object):
	def getSum(self, d, triangle, level, index):
		if level >= len(triangle):
			return 0
		if (level, index) in d:
			return d[(level, index)]

		if (level+1, index) not in d:
			d[(level+1, index)] = self.getSum(d, triangle, level+1, index)
		sum1 = d[(level+1, index)]

		if (level+1, index+1) not in d:
			d[(level+1, index+1)] = self.getSum(d, triangle, level+1, index+1)
		sum2 = d[(level+1, index+1)]

		return min(sum1+triangle[level][index], sum2+triangle[level][index])

	def minimumTotal(self, triangle):
		return self.getSum({}, triangle, 0, 0)
```
**Solution2:** using a bottom-up dynamic programming method, which can reduce the space complexity down to $$O(n)$$. Very nice and clever solution!!
```python
class Solution(object):
	def minimumTotal(self, triangle):
		dp = triangle[-1]
		for level in reversed(range(len(triangle)-1)):
			for i in range(level+1):
				dp[i] = min(dp[i],dp[i+1]) + triangle[level][i]
		return dp[0]
```
--------------------------------------------------------
### 121. Best Time to Buy and Sell Stock <a name="121"></a>
```python
class Solution(object):
	def maxProfit(self, prices):
		if len(prices) == 0:
			return 0
		maxProfit = 0
		preMin = prices[0]

		for i in prices:
			if i - preMin > maxProfit:
				maxProfit = i - preMin
			preMin = min(preMin, i)
		return maxProfit
```
--------------------------------------------------------
### 122. Best Time to Buy and Sell Stock II <a name="122"></a>
```python
class Solution(object):
	def maxProfit(self, price):
		sum = 0
		for i in range(1,len(price)):
			if price[i] > price[i-1]:
				sum += price[i] - price[i-1]
		return sum
```
--------------------------------------------------------
### 123. Best Time to Buy and Sell Stock III <a name="123"></a>
**Solution:** my solution is to main two array, one for maximum profit from zero to i, one fro maximum profit from i to end. The time complexity and space complexity are both $$O(n)$$. Solution with constant space complexity are available in the discussion forum.
```python
class Solution(object):
	def maxProfit(self, price):
		if len(price) == 0:
			return 0

		profit_zero2i = [0]
		profit_i2end = [0]

		max_zero2i = 0
		prev1 = price[0]

		max_i2end = 0
		prev2 = price[-1]

		for i in range(len(price)):
			if price[i] - prev1 > max_zero2i:
				max_zero2i = price[i] - prev1
			prev1 = min(prev1, price[i])
			profit_zero2i.append(max_zero2i)

			if price[~i] - prev2 < max_i2end:
				max_i2end = price[~i] - prev2
			prev2 = max(prev2, price[~i])
			profit_i2end.append(-max_i2end)

		maxProfit = 0
		for i in range(len(profit_zero2i)):
			maxProfit = max(maxProfit, profit_zero2i[i] + profit_i2end[~i])

		return maxProfit
```
--------------------------------------------------------
### 125. Valid Palindrome <a name="125"></a>
```python
class Solution(object):
	def isCharacter(self, x):
		if (x >= 'a' and x <= 'z') or (x >='A' and x<='Z') or (x >= '0' and x <= '9'):
			return True
		return False
	def isPalindrome(self, s):
		l, r = 0, len(s)-1
		while l < r:
			while l < r and not self.isCharacter(s[l]):
				l += 1

			while l < r and not self.isCharacter(s[r]):
				r -= 1

			if s[l].lower() != s[r].lower():
				return False
			l += 1
			r -= 1
		return True
```
--------------------------------------------------------
### 129. Sum Root to Leaf Numbers <a name="129"></a>
```python
class Solution(object):
	ans = 0
	def dfs(self, root, prev):
		if root.left == None and root.right == None:
			self.ans += int(prev+str(root.val))
		if root.left != None:
			self.dfs(root.left, prev+str(root.val))
		if root.right != None:
			self.dfs(root.right, prev+str(root.val))

	def sumNumbers(self, root):
		self.ans = 0
		if root != None:
			self.dfs(root)
		return self.ans
```
--------------------------------------------------------
### 130. Surrounded Regions <a name="130"></a>
```python
class Solution(object):
	def dfs(self, board, x, y):
		if board[x][y] != 'O':
			return
		board[x][y] = 'M'
		if x > 0:
			self.dfs(board, x-1, y)
		if x < len(board)-1:
			self.dfs(board, x+1, y)
		if y > 0:
			self.dfs(board, x, y-1)
		if y < len(board[0])-1:
			self.dfs(board, x, y+1)

	def solve(self, board):
		m, n = len(board), 0
		if m != 0:
			n = len(board[0])

		for j in range(n):
			self.dfs(board, 0, j)
			self.dfs(board, m-1, j)
		for i in range(m):
			self.dfs(board, i, 0)
			self.dfs(board, i, n-1)

		for i in range(m):
			for j in range(n):
				if board[i][j] == 'O':
					board[i][j] = 'X'
		for i in range(m):
			for j in range(n):
				if board[i][j] == 'M':
					board[i][j] = 'O'
		return board
```
--------------------------------------------------------
### 134. Gas Station <a name="134"></a>
**Solution 1:** if the sum of the array is no less than zero, it must have a solution. One of the solutions is to check the start index of maximum subarray.
```python
class Solution(object):
	def canCompleteCircuit(self, gas, cost):
		budget = [gas[i]-cost[i] for i in range(len(gas))] * 2

		if sum(budget) < 0:
			return -1

		start = 0
		max_sum = budget[0]
		max_index = 0
		temp = 0

		for i in range(len(budget)):
			temp += budget[i]
			if temp > max_sum:
				max_sum = temp
				max_index = start
			if temp < 0:
				temp = 0
				start = i+1

		return max_index
```
**Solution 2:** one more efficient solution is as follows.
```python
class Solution(object):
	def canCompleteCircuit(self, gas, cost):
		start, end = len(gas)-1, 0
		sum = gas[start] - cost[start]
		while start > end:
			if sum > 0:
				sum += gas[end] - cost[end]
				end += 1
			else:
				start -= 1
				sum += gas[start] - cost[start]
		if sum < 0:
			return -1
		return start
```
--------------------------------------------------------
### 136. Single Number <a name="136"></a>
**Solution:** use bit operation.
```python
class Solution(object):
	def singleNumber(self, nums):
		if len(nums) > 0:
			ans = nums[0]
			for i in range(1, len(nums)):
				ans ^= nums[i]
			return ans
```
--------------------------------------------------------
### 137. Signel Number <a name="137"></a>
**Solution:** a $$O(n)$$ time complexity algorithm without extra memory is possible using some complicated bit manipulations. Please refer to the discussion.
```python
class Solution(object):
	def singleNumber(self, nums):
		d = {}
		for i in nums:
			if i not in d:
				d[i] = 1
			else:
				d[i] += 1
		for key, value in d.items():
			if value == 1:
				return key
```
--------------------------------------------------------
### 152. Maximum Product Subarray <a name="152"></a>
**Solution:** a $$O(n)$$ solution.
```python
class Solution(object):
	def maxProduct(self, nums):
		ans = nums[0]
		max_pos = ans
		min_neg = ans

		for i in range(1, len(nums)):
			if nums[i] < 0:
				max_pos, min_neg = min_neg, max_pos
			max_pos = max(nums[i], max_pos * nums[i])
			min_neg = min(nums[i], min_neg * nums[i])
			ans = max(ans, max_pos)
		return ans
```
--------------------------------------------------------
### 165. Compare Version Numbers <a name="165"></a>
**Solution:** be careful about the trailing zeroes.
```python
class Solution(object):
	def compareVersion(self, version1, version2):
		s1 = [int(i) for i in version1.split('.')]
		s2 = [int(i) for i in version2.split('.')]

		while len(s1) > 0 and s1[-1] == 0:
			s1.pop()
		while len(s2) > 0 and s2[-1] == 0:
			s2.pop()

		for i in range(min(len(s1), len(s2))):
			if s1[i] > s2[i]:
				return 1
			elif s1[i] < s2[i]:
				return -1

		if len(s1) > len(s2):
			return 1
		elif len(s1) < len(s2):
			return -1
		else:
			return 0
```
--------------------------------------------------------
### 166. Fraction to Recurring Decimal <a name="166"></a>
```python
class Solution(object):
	def fractionToDecimal(self, numerator, denominator):
		if numerator == 0:
			return '0'

		sign = (numerator>0) == (denominator>0)
		numerator, denominator = abs(numerator), abs(denominator)
		ans1, ans2 = str(numerator // denominator), ""
		rem = numerator % denominator

		if rem != 0:
			ans1 = ans1 + '.'
			numerator = rem
			d = {}
			i = 0

			while numerator != 0:
				d[numerator] = i
				numerator *= 10

				quo = numerator // denominator
				rem = numerator % denominator
				ans2 = ans2 + str(quo)

				if rem in d:
					ans2 = ans2[:d[rem]] + '(' + ans2[d[rem]:] + ')'
					break
				else:
					numerator = rem
				i += 1

		return ans1+ans2 if sign else '-'+ans1+ans2
```
--------------------------------------------------------
### 167. Two Sum II - Input array is sorted <a name="167"></a>

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
### 168. Excel Sheet Column Title <a name="168"></a>
```python
class Solution(object):
	def convertToTitle(self, n):
		ans = ""
		while n != 0:
			rem = (n-1) % 26
			ans = chr(ord('A')+rem) + ans
			n = (n-1) // 26
		return ans
```
--------------------------------------------------------
### 169. Majority Element <a name="169"></a>

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
### 171. Excel Sheet Column Number <a name="171"></a>
```python
class Solution(object):
	def titleToNumber(self, s):
		ans = 0
		for i in range(len(s)):
			ans *= 26
			ans += ord(s[i]) - ord('A') + 1
		return ans
```
--------------------------------------------------------
### 172. Factorial Trailing Zeroes <a name="172"></a>
```python
class Solution(object):
	def trailingZeroes(self, n):
		ans = 0
		i = 1
		while n >= 5**i:
			ans += n // (5**i)
			i += 1
		return ans
```
--------------------------------------------------------
### 179. Largest Number <a name="179"></a>
**Solution:** use customized compare function
```python
class Compare(str):
	def __lt__(x,y):
		return x+y > y+x

class Solution(object):
	def largestNumber(self, nums):
		str_num = [str(i) for i in nums]
		ans = "".join(sorted(str_num, key=Compare))
		return '0' if ans[0]=='0' else ans
```
--------------------------------------------------------
### 198. House Robber <a name="198"></a>
**Solution:** simple DP.
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = len(nums)
        if l==0:
        	return 0
        if l==1:
        	return nums[0]
        if l==2:
        	return max(nums[0],nums[1])
        f = [0]*(l+1)
        f[1] = nums[0]
        f[2] = max(nums[1],nums[0])
        for i in range(3,l+1):
        	f[i] = max(f[i-1], f[i-2]+nums[i-1])
        return f[-1]
```
--------------------------------------------------------

### 213. House Robber II <a name="213"></a>
**Solution:** similar to House Robber I, only consider either robbing house 1 or not. The space complexity can be easily reduced to $$O(1)$$.
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = len(nums)
        if l==0:
        	return 0
        if l==1:
        	return nums[0]
        if l==2:
        	return max(nums[0],nums[1])
        f0 = [0]*(l+1) # not rob 1
        f1 = [0]*(l+1) # rob 1

        f0[1], f0[2] = 0, nums[1]
        f1[1], f1[2] = nums[0], nums[0]

        for i in range(3,l):
        	f0[i] = max(f0[i-1], f0[i-2]+nums[i-1])
        	f1[i] = max(f1[i-1], f1[i-2]+nums[i-1])

        max1 = max(f0[l-1], f0[l-2]+nums[l-1])
        return max(max1, f1[l-1])
```
--------------------------------------------------------
### 217. Contains Duplicate <a name="217"></a>
```python
class Solution(object):
	def containsDuplicate(self, nums):
		d = {}
		for i in nums:
			if i not in d:
				d[i] = 1
			else:
				return True
		return False
```
--------------------------------------------------------
### 219. Contains Duplicate <a name="219"></a>
```python
class Solution(object):
	def containsNearbyDuplicate(self, nums, k):
		d = {}
		for i in range(len(nums)):
			if nums[i] in d and i - d[nums[i]] <= k:
					return True
			d[nums[i]] = i
		return False
```
--------------------------------------------------------

### 242. Valid Anagram <a name="242"></a>

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

### 278. First Bad Version <a name="278"></a>

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

### 349. Intersection of Two Arrays

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

### 350. Intersection of Two Arrays II <a name="350"></a>

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
### 415. Add Strings <a name='415'></a>
```python
class Solution(object):
	def addStrings(self, num1, num2):
		if len(num1) < len(num2):
			num1, num2 = num2, num1
		ans = [0] * (len(num1) + 1)
		for i in range(1, len(num2)+1):
			sum = ans[-i] + ord(num1[-i]) + ord(num2[-i]) - 2*ord('0')
			ans[-i] = sum % 10
			ans[-i-1] = sum//10
		for i in range(len(num2)+1, len(num1)+1):
			sum = ans[-i] + ord(num1[-i]) - ord('0')
			ans[-i] = sum % 10
			ans[-i-1] = sum//10
		ans = ''.join([str(i) for i in ans])
		if ans[0] != '0':
			return ans
		else:
			return ans[1:]
```
--------------------------------------------------------

### 687. Longest Univalue Path <a name="687"></a>

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
