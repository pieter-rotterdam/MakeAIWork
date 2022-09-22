import unittest
from matrix import Matrix

class TestMatrix(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print('setupClass')

    vectorList1 = ([1, 2, 3], [4, 5, 6])
    vectorList2 = ([1, 1, 1], [1, 1, 1])
    
    def test_(self,):
        print('test_addition')
        self.assertEqual(matrix2.vectorList, ([1, 1, 1], [1, 1, 1]))

# self.assertEqual(vectorList3 = [self.addVectors(v1, v2) for (v1, v2) in zip(vectorList1, vectorList2)], Matrix3 [[2, 3, 4], [5, 6, 7]])

#     def test_annualSalary(self):
#         print('test_apply_raise')
#         self.employee1.annualSalary
#         self.employee2.annualSalary

if __name__ == '__main__':
     unittest.main()
