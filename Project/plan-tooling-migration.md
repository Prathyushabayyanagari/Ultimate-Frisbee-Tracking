# UltimatePlayerTracker Tooling Migration Plan

## Project Overview

This plan outlines the migration from Poetry to Pixi and from flake8/black to ruff for the UltimatePlayerTracker project. This migration will modernize the development tooling and provide better performance, cross-platform support, and simplified dependency management.

## Current Tooling (TO BE REPLACED)

### Package Management
- **Poetry**: Dependency management and virtual environments
- **pyproject.toml**: Project configuration
- **poetry.lock**: Dependency lock file

### Code Quality Tools
- **black**: Code formatting (v21.9b0)
- **flake8**: Linting (v4.0.1)
- **pycodestyle**: Style checking (v2.8.0)
- **mypy**: Type checking (v0.910)

### Current Commands
```bash
# Format code
poetry run black .

# Run linting
poetry run flake8 .

# Run type checking
poetry run mypy .

# Run all quality checks together
poetry run black . && poetry run flake8 . && poetry run mypy .
```

## Target Tooling (TO BE IMPLEMENTED)

### Package Management
- **Pixi**: Modern dependency management with Conda + PyPI support
- **pyproject.toml**: Enhanced project configuration with Pixi sections
- **pixi.lock**: Dependency lock file

### Code Quality Tools
- **ruff**: Combined linting and formatting (10-100x faster than black/flake8)
- **mypy**: Type checking (unchanged)

### Target Commands
```bash
# Format code
pixi run ruff format .

# Run linting
pixi run ruff check .

# Fix auto-fixable issues
pixi run ruff check --fix .

# Run type checking
pixi run mypy .

# Run all quality checks together
pixi run ruff check --fix . && pixi run ruff format . && pixi run mypy .
```

---

## Phase 1: Setup and Planning (1-2 days)

### ☐ Task 1.1: Install Pixi and Initialize Project
**Estimated Time:** 2-4 hours  
**Requirements:**
- [ ] Install Pixi globally: `curl -fsSL https://pixi.sh/install.sh | bash`
- [ ] Initialize Pixi in project: `pixi init --format pyproject`
- [ ] Test basic Pixi functionality: `pixi run python --version`
- [ ] Verify Pixi shell activation: `pixi shell`

**Deliverable:**
- Working Pixi environment in project directory
- pixi.toml file created

### ☐ Task 1.2: Research Current Dependencies
**Estimated Time:** 1-2 hours  
**Requirements:**
- [ ] Analyze current pyproject.toml dependencies
- [ ] Check which dependencies are available via Conda vs PyPI
- [ ] Identify any platform-specific dependencies
- [ ] Document dependency migration strategy

**Deliverable:**
- Dependency migration report
- List of Conda vs PyPI dependencies

---

## Phase 2: Pixi Migration (2-3 days)

### ☐ Task 2.1: Update pyproject.toml for Pixi
**Estimated Time:** 4-6 hours  
**Requirements:**
- [ ] Add Pixi configuration sections to pyproject.toml
- [ ] Configure channels (conda-forge, etc.)
- [ ] Set up target platforms
- [ ] Migrate dependencies from Poetry to Pixi format
- [ ] Add Python version constraint

**Code Changes Required:**
```toml
# ADD to pyproject.toml:
[project]
name = "ultimate-player-tracker"
version = "0.1.0"
requires-python = ">=3.8"

[tool.pixi.project]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

[tool.pixi.dependencies]
python = ">=3.8"
# Conda dependencies go here

[tool.pixi.pypi-dependencies]
# PyPI dependencies go here
```

### ☐ Task 2.2: Migrate Dependencies
**Estimated Time:** 3-4 hours  
**Requirements:**
- [ ] Separate Conda vs PyPI dependencies
- [ ] Update TensorFlow dependencies (consider GPU variants)
- [ ] Add computer vision dependencies
- [ ] Migrate development dependencies
- [ ] Test dependency installation

**Example Migration:**
```toml
# FROM Poetry:
tensorflow = "2.3.0"
opencv-python = "4.1.*"
numpy = "^1.21.0"

# TO Pixi:
[tool.pixi.dependencies]
python = ">=3.8"
numpy = ">=1.21.0"
# Use conda-forge versions where available

[tool.pixi.pypi-dependencies]
tensorflow-cpu = "~=2.3.0"  # or tensorflow-gpu
opencv-python = "~=4.1.0"
```

### ☐ Task 2.3: Create Pixi Tasks
**Estimated Time:** 1-2 hours  
**Requirements:**
- [ ] Define common tasks in pixi.toml
- [ ] Create development environment tasks
- [ ] Add testing and linting tasks
- [ ] Create application running tasks

**Task Configuration:**
```toml
[tool.pixi.tasks]
format = "ruff format ."
lint = "ruff check ."
lint-fix = "ruff check --fix ."
type-check = "mypy ."
check-all = "ruff check --fix . && ruff format . && mypy ."
run-tracker = "python object_tracker.py --video ./data/video/demo.mp4"
```

---

## Phase 3: Ruff Migration (1-2 days)

### ☐ Task 3.1: Configure Ruff
**Estimated Time:** 2-3 hours  
**Requirements:**
- [ ] Add ruff configuration to pyproject.toml
- [ ] Configure line length and formatting style
- [ ] Set up linting rules (replace flake8 rules)
- [ ] Configure per-file ignores
- [ ] Add mypy compatibility

**Ruff Configuration:**
```toml
[tool.ruff]
line-length = 88
target-version = "py38"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C90", "UP"]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
```

### ☐ Task 3.2: Replace Black/Flake8 with Ruff
**Estimated Time:** 2-3 hours  
**Requirements:**
- [ ] Remove black, flake8, pycodestyle from dependencies
- [ ] Add ruff to development dependencies
- [ ] Update all formatting and linting commands
- [ ] Test ruff formatting on existing codebase
- [ ] Verify ruff catches same issues as flake8

**Dependency Changes:**
```toml
# REMOVE:
black = "21.9b0"
flake8 = "4.0.1"
pycodestyle = "2.8.0"

# ADD:
ruff = "*"
```

---

## Phase 4: Validation and Testing (1-2 days)

### ☐ Task 4.1: Test Environment Setup
**Estimated Time:** 2-3 hours  
**Requirements:**
- [ ] Clean test: Remove .venv, poetry.lock, pixi.lock
- [ ] Fresh install: `pixi install`
- [ ] Test all tasks: `pixi run check-all`
- [ ] Verify application still runs
- [ ] Test on different platforms if available

### ☐ Task 4.2: Update Documentation
**Estimated Time:** 1-2 hours  
**Requirements:**
- [ ] Update AGENTS.md with new commands
- [ ] Update README.md with Pixi instructions
- [ ] Document any new environment variables
- [ ] Create troubleshooting section for Pixi

**AGENTS.md Updates:**
```markdown
### Code Quality Tools
The project uses the following development dependencies:
- **ruff**: Combined linting and formatting
- **mypy**: Type checking

```bash
# Format code
pixi run ruff format .

# Run linting
pixi run ruff check .

# Fix auto-fixable issues
pixi run ruff check --fix .

# Run type checking
pixi run mypy .

# Run all quality checks together
pixi run ruff check --fix . && pixi run ruff format . && pixi run mypy .
```

### ☐ Task 4.3: Performance Benchmarking
**Estimated Time:** 1-2 hours  
**Requirements:**
- [ ] Benchmark formatting speed (ruff vs black)
- [ ] Benchmark linting speed (ruff vs flake8)
- [ ] Document performance improvements
- [ ] Create performance comparison report

---

## Phase 5: Cleanup (1 day)

### ☐ Task 5.1: Remove Legacy Tooling
**Estimated Time:** 1-2 hours  
**Requirements:**
- [ ] Remove poetry.lock file
- [ ] Remove any black/flask8 configuration files
- [ ] Update CI/CD pipelines to use Pixi
- [ ] Archive old tooling documentation

### ☐ Task 5.2: Final Validation
**Estimated Time:** 2-3 hours  
**Requirements:**
- [ ] Complete end-to-end test of the application
- [ ] Verify all development tasks work
- [ ] Test on fresh developer machine (if possible)
- [ ] Create migration summary report

---

## Migration Benefits

### Performance Improvements
- **10-100x faster** linting and formatting with Ruff
- **Faster dependency resolution** with Pixi's Conda integration
- **Reduced virtual environment setup time**

### Developer Experience
- **Cross-platform compatibility** with Pixi
- **Unified tooling** (one tool for formatting + linting)
- **Better dependency management** with Conda + PyPI hybrid
- **Simplified commands** and task system

### Maintenance
- **Fewer dependencies** to maintain
- **Active development** on both Pixi and Ruff
- **Better error messages** and debugging
- **Modern Python packaging standards**

---

## Risk Mitigation

### Potential Issues
1. **Dependency Compatibility**
   - **Risk:** Some dependencies may not be available via Conda
   - **Mitigation:** Use pypi-dependencies section for problematic packages

2. **Code Formatting Differences**
   - **Risk:** Ruff may format slightly differently than Black
   - **Mitigation:** Configure Ruff to match Black style exactly

3. **Learning Curve**
   - **Risk:** Team unfamiliar with Pixi/Ruff commands
   - **Mitigation:** Comprehensive documentation and migration guide

### Rollback Plan
- Keep original pyproject.toml backup
- Document Poetry commands for emergency rollback
- Maintain parallel setup during transition period

---

## Success Criteria

### Functional Requirements
- [ ] **All existing commands work** with Pixi
- [ ] **Application runs identically** after migration
- [ ] **Code formatting maintains** exact same style
- [ ] **Linting catches same issues** as flake8

### Performance Targets
- [ ] **10x+ faster** linting than flake8
- [ ] **10x+ faster** formatting than black
- [ ] **Faster dependency installation** than Poetry
- [ ] **Reduced setup time** for new developers

### Quality Targets
- [ ] **Zero functionality regression** in main application
- [ ] **Maintained code quality standards**
- [ ] **Complete documentation** updated
- [ ] **Successful CI/CD integration**

---

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Setup | 1-2 days | Day 1 | Day 2 |
| Phase 2: Pixi Migration | 2-3 days | Day 2 | Day 5 |
| Phase 3: Ruff Migration | 1-2 days | Day 5 | Day 7 |
| Phase 4: Validation | 1-2 days | Day 7 | Day 9 |
| Phase 5: Cleanup | 1 day | Day 9 | Day 10 |

**Total Estimated Time:** 7-10 days

---

## Next Steps

1. **Review and approve this tooling migration plan**
2. **Begin Phase 1: Install Pixi and research dependencies**
3. **Execute migration phases sequentially**
4. **Validate thoroughly before proceeding with YOLO migration**
5. **Update team on new development workflow**

*This tooling migration provides a solid foundation for the subsequent YOLO migration and modernizes the development experience for UltimatePlayerTracker.*