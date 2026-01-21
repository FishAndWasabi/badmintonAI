"""Module registry for dynamic module loading and management

This module implements a registry pattern similar to MMDetection/MMPose,
allowing dynamic module registration and instantiation from configuration.

Reference: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html
"""

from typing import Dict, Type, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import inspect


@dataclass
class ModuleInfo:
    """Information about a registered module"""
    name: str
    module_class: Type
    description: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None  # Module file location


class Registry:
    """Registry for dynamically loading and managing modules
    
    This class follows the pattern used in MMDetection/MMPose frameworks.
    It supports:
    - Decorator-based registration: @registry.register_module()
    - Build from config: registry.build(cfg)
    - Automatic class name inference
    - Function registration support
    
    Example:
        # Register a module
        @registry.register_module()
        class MyTracker:
            def __init__(self, param1, param2):
                self.param1 = param1
                self.param2 = param2
        
        # Build from config
        cfg = dict(type='MyTracker', param1=1.0, param2='test')
        tracker = registry.build(cfg)
    """
    
    def __init__(self, name: str):
        """
        Initialize a registry
        
        Args:
            name: Registry name (for identification and error messages)
        """
        self._name = name
        self._module_dict: Dict[str, ModuleInfo] = {}
    
    def _register_module(self, module_class: Type, name: Optional[str] = None,
                        force: bool = False, module: Optional[str] = None) -> Type:
        """
        Register a module class
        
        Args:
            module_class: Module class or function to register
            name: Module name (if None, use class/function name)
            force: Whether to override existing registration
            module: Module location (for tracking)
        
        Returns:
            The registered class/function
        """
        if not inspect.isclass(module_class) and not inspect.isfunction(module_class):
            raise TypeError(
                f'module must be a class or function, but got {type(module_class)}'
            )
        
        module_name = name if name is not None else module_class.__name__
        
        if not force and module_name in self._module_dict:
            raise KeyError(
                f'{module_name} is already registered in {self._name}'
            )
        
        # Get module location
        if module is None:
            mod = inspect.getmodule(module_class)
            module = mod.__name__ if mod else None
        
        self._module_dict[module_name] = ModuleInfo(
            name=module_name,
            module_class=module_class,
            location=module
        )
        return module_class
    
    def register_module(self, name: Optional[str] = None, 
                       force: bool = False, module: Optional[str] = None) -> Callable:
        """
        Decorator to register a module
        
        Args:
            name: Module name (if None, use class/function name)
            force: Whether to override existing registration
            module: Module location
        
        Returns:
            Decorator function
        
        Example:
            @registry.register_module()
            class MyClass:
                pass
            
            @registry.register_module(name='custom_name')
            class AnotherClass:
                pass
        """
        def _register(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
            self._register_module(cls_or_func, name=name, force=force, module=module)
            return cls_or_func
        
        return _register
    
    def get(self, name: str) -> Union[Type, Callable]:
        """
        Get a registered module by name
        
        Args:
            name: Module name
        
        Returns:
            Registered module class or function
        
        Raises:
            KeyError: If module not found
        """
        if name not in self._module_dict:
            available = list(self._module_dict.keys())
            raise KeyError(
                f'{name} is not in the {self._name} registry. '
                f'Available modules: {available}'
            )
        return self._module_dict[name].module_class
    
    def build(self, cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
        """
        Build a module instance from config dictionary
        
        This is the core method following MMDetection/MMPose pattern.
        The config dict should contain a 'type' key specifying the module name,
        and other keys are passed as keyword arguments to the constructor.
        
        Args:
            cfg: Config dict containing 'type' key and other parameters
            default_args: Optional default arguments to merge with cfg
        
        Returns:
            Instantiated module
        
        Raises:
            TypeError: If cfg is not a dict or type is invalid
            KeyError: If 'type' key is missing or module not found
        
        Example:
            cfg = dict(type='YOLOBallTracker', conf_threshold=0.5, iou_threshold=0.45)
            tracker = registry.build(cfg)
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        
        if 'type' not in cfg:
            raise KeyError(f'cfg must contain "type" key, but got {cfg}')
        
        args = cfg.copy()
        obj_type = args.pop('type')
        
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}'
            )
        
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        
        return obj_cls(**args)
    
    def list_modules(self) -> list:
        """List all registered module names"""
        return list(self._module_dict.keys())
    
    def has(self, name: str) -> bool:
        """Check if a module is registered"""
        return name in self._module_dict
    
    def get_info(self, name: str) -> ModuleInfo:
        """Get module information"""
        if name not in self._module_dict:
            raise KeyError(f'{name} is not in the {self._name} registry')
        return self._module_dict[name]
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator: 'name' in registry"""
        return self.has(name)
    
    def __len__(self) -> int:
        """Return number of registered modules"""
        return len(self._module_dict)
    
    def __repr__(self) -> str:
        """String representation"""
        return f'{self.__class__.__name__}(name={self._name}, items={len(self._module_dict)})'


class ModuleRegistry:
    """Multi-category module registry manager
    
    Manages multiple registries for different module categories.
    Provides convenience methods for each category following MMDetection/MMPose patterns.
    """
    
    def __init__(self):
        """Initialize the multi-category registry"""
        self._registries: Dict[str, Registry] = {
            'ball_tracker': Registry('ball_tracker'),
            'pose_estimator': Registry('pose_estimator'),
            'shot_classifier': Registry('shot_classifier')
        }
    
    def get_registry(self, category: str) -> Registry:
        """Get a specific registry by category"""
        if category not in self._registries:
            available = list(self._registries.keys())
            raise ValueError(
                f"Unknown category: {category}. Available: {available}"
            )
        return self._registries[category]
    
    def register(self, category: str, name: Optional[str] = None, **kwargs) -> Callable:
        """
        Decorator to register a module in a category
        
        Args:
            category: Module category
            name: Optional module name (uses class name if None)
            **kwargs: Additional arguments passed to register_module
        
        Returns:
            Decorator function
        
        Example:
            @registry.register('ball_tracker')
            class MyTracker:
                pass
        """
        registry = self.get_registry(category)
        return registry.register_module(name=name, **kwargs)
    
    def register_ball_tracker(self, name: Optional[str] = None, **kwargs) -> Callable:
        """Convenience decorator for ball trackers"""
        return self.register('ball_tracker', name=name, **kwargs)
    
    def register_pose_estimator(self, name: Optional[str] = None, **kwargs) -> Callable:
        """Convenience decorator for pose estimators"""
        return self.register('pose_estimator', name=name, **kwargs)
    
    def register_shot_classifier(self, name: Optional[str] = None, **kwargs) -> Callable:
        """Convenience decorator for shot classifiers"""
        return self.register('shot_classifier', name=name, **kwargs)
    
    def get(self, category: str, name: str) -> Type:
        """Get a module class by category and name"""
        registry = self.get_registry(category)
        return registry.get(name)
    
    def get_ball_tracker(self, name: str) -> Type:
        """Get ball tracker class"""
        return self.get('ball_tracker', name)
    
    def get_pose_estimator(self, name: str) -> Type:
        """Get pose estimator class"""
        return self.get('pose_estimator', name)
    
    def get_shot_classifier(self, name: str) -> Type:
        """Get shot classifier class"""
        return self.get('shot_classifier', name)
    
    def build(self, category: str, cfg: Dict[str, Any], 
              default_args: Optional[Dict[str, Any]] = None) -> Any:
        """
        Build a module instance from config (MMDetection/MMPose style)
        
        Args:
            category: Module category
            cfg: Config dict with 'type' key and other parameters
            default_args: Optional default arguments to merge
        
        Returns:
            Instantiated module
        
        Example:
            cfg = dict(type='yolo_tracker', conf_threshold=0.5)
            tracker = registry.build('ball_tracker', cfg)
        """
        registry = self.get_registry(category)
        return registry.build(cfg, default_args=default_args)
    
    def build_ball_tracker(self, cfg: Dict[str, Any], 
                          default_args: Optional[Dict[str, Any]] = None) -> Any:
        """Build ball tracker from config"""
        return self.build('ball_tracker', cfg, default_args=default_args)
    
    def build_pose_estimator(self, cfg: Dict[str, Any],
                            default_args: Optional[Dict[str, Any]] = None) -> Any:
        """Build pose estimator from config"""
        return self.build('pose_estimator', cfg, default_args=default_args)
    
    def build_shot_classifier(self, cfg: Dict[str, Any],
                             default_args: Optional[Dict[str, Any]] = None) -> Any:
        """Build shot classifier from config"""
        return self.build('shot_classifier', cfg, default_args=default_args)
    
    def list_modules(self, category: Optional[str] = None) -> Dict[str, list]:
        """List all registered modules"""
        if category:
            return {category: self.get_registry(category).list_modules()}
        return {cat: reg.list_modules() for cat, reg in self._registries.items()}
    
    def has(self, category: str, name: str) -> bool:
        """Check if a module is registered"""
        return self.get_registry(category).has(name)
    
    def get_info(self, category: str, name: str) -> ModuleInfo:
        """Get module information"""
        return self.get_registry(category).get_info(name)


# Global registry instance
registry = ModuleRegistry()

# Convenience access to individual registries (MMDetection/MMPose style)
BALL_TRACKERS = registry.get_registry('ball_tracker')
POSE_ESTIMATORS = registry.get_registry('pose_estimator')
SHOT_CLASSIFIERS = registry.get_registry('shot_classifier')


