import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:02:07.036854
# Source Brief: brief_00221.md
# Brief Index: 221
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class Branch:
    """A node in the fractal tree, representing a single branch."""
    def __init__(self, parent, start_pos, angle, length, width, depth, color):
        self.parent = parent
        self.start_pos = start_pos
        self.angle = angle
        self.length = length
        self.width = int(max(1, width))
        self.depth = depth
        self.color = color
        
        self.end_pos = self.start_pos + pygame.Vector2(0, -self.length).rotate(self.angle)
        self.children = []
        self.is_pruned = False

    def is_leaf(self):
        return not self.children

class Tree:
    """Manages a single fractal tree's state, growth, and rendering."""
    def __init__(self, root_pos, config):
        self.root_pos = root_pos
        self.config = config
        self.root = Branch(
            parent=None,
            start_pos=self.root_pos,
            angle=0,
            length=config['initial_length'],
            width=config['initial_width'],
            depth=0,
            color=config['color_branch']
        )

    def get_all_nodes(self, only_active=False):
        """Traverse the tree and return a flat list of all branch nodes."""
        nodes = []
        q = deque([self.root])
        while q:
            node = q.popleft()
            if only_active and node.is_pruned:
                continue
            nodes.append(node)
            for child in node.children:
                q.append(child)
        return nodes

    def grow(self):
        """Add new branches to a random leaf node."""
        active_nodes = self.get_all_nodes(only_active=True)
        if len(active_nodes) >= self.config['max_branches']:
            return

        leaf_nodes = [node for node in active_nodes if node.is_leaf()]
        if not leaf_nodes:
            return

        parent_branch = random.choice(leaf_nodes)
        
        new_length = parent_branch.length * self.config['length_decay']
        new_width = parent_branch.width * self.config['width_decay']
        
        if new_length < 2 or new_width < 1:
            return

        # Create two new branches
        for angle_mod in [-1, 1]:
            new_angle = parent_branch.angle + self.config['angle_delta'] * angle_mod
            child = Branch(
                parent=parent_branch,
                start_pos=parent_branch.end_pos,
                angle=new_angle,
                length=new_length,
                width=new_width,
                depth=parent_branch.depth + 1,
                color=self.config['color_branch']
            )
            parent_branch.children.append(child)

    def prune(self):
        """Mark a random, prunable branch as pruned."""
        prunable_nodes = [node for node in self.get_all_nodes(only_active=True) if node.parent is not None]
        
        if not prunable_nodes:
            return None

        node_to_prune = random.choice(prunable_nodes)
        node_to_prune.is_pruned = True
        return node_to_prune

    def count_active_branches(self):
        return len(self.get_all_nodes(only_active=True))

    def draw(self, surface):
        """Draw the entire tree."""
        for node in self.get_all_nodes(only_active=False):
            color = self.config['color_pruned'] if node.is_pruned else node.color
            if node.width <= 1:
                pygame.draw.aaline(surface, color, node.start_pos, node.end_pos)
            else:
                pygame.draw.line(surface, color, node.start_pos, node.end_pos, int(node.width))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Prune rapidly growing fractal trees to maintain a target number of branches before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys or Shift to select a tree (← for right, ↓/Shift for middle, ↑/→ for left). "
        "Press Space to prune."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        
        # Colors and Config
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_BRANCH = (40, 200, 120)
        self.COLOR_PRUNED = (60, 70, 80)
        self.COLOR_UI = (220, 220, 240)
        self.COLOR_SELECT = (255, 255, 100)
        self.COLOR_PARTICLE = (255, 80, 80)

        self.TREE_CONFIG = {
            'initial_length': 70,
            'initial_width': 8,
            'length_decay': 0.8,
            'width_decay': 0.85,
            'angle_delta': 25,
            'max_branches': 100,
            'color_branch': self.COLOR_BRANCH,
            'color_pruned': self.COLOR_PRUNED,
        }
        self.TARGET_BRANCHES = 7
        self.FPS = 30
        self.MAX_SECONDS = 60
        self.MAX_STEPS = self.MAX_SECONDS * self.FPS
        
        # Initialize state variables
        self.trees = []
        self.particles = []
        self.selected_tree_idx = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.selected_tree_idx = 0
        self.particles = []

        tree_positions = [
            pygame.Vector2(self.WIDTH * 0.2, self.HEIGHT - 20),
            pygame.Vector2(self.WIDTH * 0.5, self.HEIGHT - 20),
            pygame.Vector2(self.WIDTH * 0.8, self.HEIGHT - 20)
        ]
        self.trees = [Tree(pos, self.TREE_CONFIG) for pos in tree_positions]

        # Pre-grow trees to a playable state
        for _ in range(20):
            for tree in self.trees:
                tree.grow()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # 1. Handle player actions
            prune_reward, prune_penalty = self._handle_actions(action)
            reward += prune_reward
            
            # 2. Update game state
            self._update_game_state()
            
            # 3. Calculate continuous rewards
            reward += self._calculate_step_reward()
            
            # 4. Check for termination
            terminated, terminal_reward = self._check_termination(prune_penalty)
            reward += terminal_reward
            self.game_over = terminated
        
        self.steps += 1
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        prune_reward = 0
        prune_penalty = False

        # --- Tree Selection ---
        if movement in [1, 4]:  # Up or Right -> selects tree 0 (left)
            self.selected_tree_idx = 0
        elif movement == 2 or shift_held:  # Down or Shift -> selects tree 1 (middle)
            self.selected_tree_idx = 1
        elif movement == 3:  # Left -> selects tree 2 (right)
            self.selected_tree_idx = 2
        
        # --- Pruning ---
        if space_held:
            # sfx: a sharp 'snip' sound
            target_tree = self.trees[self.selected_tree_idx]
            count_before = target_tree.count_active_branches()
            
            if count_before > self.TARGET_BRANCHES:
                pruned_branch = target_tree.prune()
                if pruned_branch:
                    prune_reward = 1.0  # Reward for successful pruning
                    self._create_particles(pruned_branch.start_pos)
            else: # Pruning when at or below target is an immediate failure
                pruned_branch = target_tree.prune()
                if pruned_branch:
                    prune_penalty = True
                    self._create_particles(pruned_branch.start_pos)
        
        return prune_reward, prune_penalty

    def _update_game_state(self):
        # Grow trees periodically
        if self.steps % (self.FPS // 5) == 0: # Grow 5 times per second
            for tree in self.trees:
                tree.grow()

        # Update timer
        self.timer -= 1
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _calculate_step_reward(self):
        # Continuous reward for maintaining trees above target
        if all(tree.count_active_branches() >= self.TARGET_BRANCHES for tree in self.trees):
            return 0.1
        return 0

    def _check_termination(self, prune_penalty):
        # Failure by over-pruning
        if prune_penalty:
            # sfx: a sad, 'breaking' sound
            return True, -100.0

        # Failure by timer running out with incorrect counts
        if self.timer <= 0:
            branch_counts = [tree.count_active_branches() for tree in self.trees]
            if all(count == self.TARGET_BRANCHES for count in branch_counts):
                # sfx: a triumphant 'chime' or 'success' sound
                return True, 100.0  # Victory
            else:
                # sfx: a muted 'fail' buzzer
                return True, 0.0 # Failure by timeout
        
        return False, 0.0

    def _create_particles(self, pos):
        for _ in range(15):
            angle = random.uniform(0, 360)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(10, 20),
                'color': self.COLOR_PARTICLE
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw selection highlight
        selected_tree = self.trees[self.selected_tree_idx]
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        radius = int(25 + pulse * 5)
        alpha = int(50 + pulse * 40)
        pygame.gfxdraw.filled_circle(self.screen, int(selected_tree.root_pos.x), int(selected_tree.root_pos.y), radius, (*self.COLOR_SELECT, alpha))
        pygame.gfxdraw.aacircle(self.screen, int(selected_tree.root_pos.x), int(selected_tree.root_pos.y), radius, (*self.COLOR_SELECT, alpha+50))

        # Draw trees
        for tree in self.trees:
            tree.draw(self.screen)
            
        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / 20
            radius = int(life_ratio * 3)
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], radius)

    def _render_ui(self):
        # Display timer
        seconds_left = max(0, self.timer // self.FPS)
        minutes = seconds_left // 60
        seconds = seconds_left % 60
        timer_text = f"{minutes:02}:{seconds:02}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Display total score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Display branch counts
        for i, tree in enumerate(self.trees):
            count = tree.count_active_branches()
            color = self.COLOR_UI
            if count == self.TARGET_BRANCHES:
                color = (100, 255, 100) # Green for correct
            elif count < self.TARGET_BRANCHES:
                color = (255, 100, 100) # Red for too few
            
            count_text = f"{count}"
            count_surf = self.font_large.render(count_text, True, color)
            pos_x = int(tree.root_pos.x - count_surf.get_width() / 2)
            pos_y = int(tree.root_pos.y + 5)
            self.screen.blit(count_surf, (pos_x, pos_y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer_seconds": max(0, self.timer // self.FPS),
            "tree_1_branches": self.trees[0].count_active_branches(),
            "tree_2_branches": self.trees[1].count_active_branches(),
            "tree_3_branches": self.trees[2].count_active_branches(),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # a bit of a hack to re-init with video
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    # Game loop
    running = True
    display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        # --- Pygame event handling ---
        pygame_events = pygame.event.get()
        for event in pygame_events:
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

        # --- Map keyboard to action space ---
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        # The observation is already what we want to display
        # Pygame uses (width, height) but numpy uses (height, width)
        # We need to transpose it back for pygame display
        surf_to_display = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surf.blit(surf_to_display, (0, 0))
            
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Info: {info}")
            obs, info = env.reset()
            
        env.clock.tick(env.FPS)
        
    env.close()