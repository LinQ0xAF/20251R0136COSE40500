from StackInspector import StackInspector

class StackInspectorDemo(StackInspector):
    def callee(self) -> None:
        func = self.caller_function()
        assert func.__name__ == 'test'
        print(func)

    def caller(self) -> None:
        self.callee()
        
def test() -> None:
    demo = StackInspectorDemo()
    demo.caller()

test()