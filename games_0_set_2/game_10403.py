import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:25:01.818864
# Source Brief: brief_00403.md
# Brief Index: 403
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    GameEnv: A Quantum Labyrinth Puzzle Roguelike
    
    Navigate a procedurally generated labyrinth by teleporting blocks to trigger 
    chain reactions and unlock the path to the exit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated labyrinth by teleporting blocks to trigger "
        "chain reactions and unlock the path to the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move and press space to teleport the block in front of you."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_BLOCK_BLUE = (50, 150, 255)
    COLOR_BLOCK_GREEN = (50, 255, 150)
    COLOR_BLOCK_RED = (255, 80, 80)
    COLOR_BLOCK_YELLOW = (255, 255, 80)
    COLOR_EXIT_1 = (200, 80, 255)
    COLOR_EXIT_2 = (255, 150, 255)
    COLOR_TELEPORT_TARGET = (255, 255, 255, 100) # RGBA
    COLOR_TEXT = (220, 220, 220)

    # Block Types
    BLOCK_EMPTY = 0
    BLOCK_BLUE = 1  # Standard movable
    BLOCK_GREEN = 2 # Explodes on landing
    BLOCK_RED = 3   # Immovable
    BLOCK_YELLOW = 4 # Not used in this version

    # Game Parameters
    MAX_STEPS = 2500
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TELEPORT_DISTANCE = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        self.player_pos = None
        self.exit_pos = None
        self.grid = None
        self.grid_width = 0
        self.grid_height = 0
        self.tile_size = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self.facing_direction = (0, -1) # Start facing up
        self.particles = []
        self.space_was_held = False

        # This call will also perform the first reset
        # self.validate_implementation() # Commented out for standard use

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.space_was_held = False
        self.facing_direction = (0, -1)
        
        if options and 'level' in options:
            self.level = options.get('level', 1)
        else:
            self.level = 1
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        
        dist_before = self._manhattan_distance(self.player_pos, self.exit_pos)

        # 1. Handle Movement
        moved = self._move_player(movement)
        if moved:
            dist_after = self._manhattan_distance(self.player_pos, self.exit_pos)
            reward += (dist_before - dist_after) * 0.1 # Reward for getting closer
        
        # 2. Handle Teleport Action (on press, not hold)
        if space_held and not self.space_was_held:
            teleport_reward = self._teleport_block()
            reward += teleport_reward
        self.space_was_held = space_held
        
        # 3. Handle Shift Action (placeholder)
        if shift_held:
            # Placeholder for future ability cycling
            pass
            
        # 4. Update Game State
        self._update_particles()
        
        # 5. Check Termination Conditions
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += reward
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        # self.level += 1 # Level is managed externally or in reset options
        
        # Scale difficulty
        base_size = 10
        size_increase = ((self.level -1) // 5) * 2
        self.grid_width = base_size + size_increase
        self.grid_height = base_size + size_increase
        
        self.tile_size = min(
            (self.SCREEN_WIDTH - 40) // self.grid_width, 
            (self.SCREEN_HEIGHT - 80) // self.grid_height
        )
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width * self.tile_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height * self.tile_size) // 2 + 20

        # Generation loop to ensure solvability
        while True:
            self.grid = np.full((self.grid_width, self.grid_height), self.BLOCK_EMPTY, dtype=int)
            
            # Place border
            self.grid[0, :] = self.BLOCK_RED
            self.grid[-1, :] = self.BLOCK_RED
            self.grid[:, 0] = self.BLOCK_RED
            self.grid[:, -1] = self.BLOCK_RED

            # Place player and exit
            self.player_pos = (self.np_random.integers(1, self.grid_width-1), self.grid_height-2)
            self.exit_pos = (self.np_random.integers(1, self.grid_width-1), 1)

            # Place blocks
            num_blocks = int(self.grid_width * self.grid_height * 0.3)
            for _ in range(num_blocks):
                x, y = self.np_random.integers(1, self.grid_width-1), self.np_random.integers(1, self.grid_height-1)
                if (x, y) != self.player_pos and (x, y) != self.exit_pos:
                    block_type = self.BLOCK_BLUE
                    if self.level >= 5 and self.np_random.random() < 0.2:
                        block_type = self.BLOCK_GREEN
                    if self.level >= 10 and self.np_random.random() < 0.1:
                        block_type = self.BLOCK_RED
                    self.grid[x, y] = block_type
            
            # Ensure player start and exit are clear
            self.grid[self.player_pos] = self.BLOCK_EMPTY
            self.grid[self.exit_pos] = self.BLOCK_EMPTY
            
            if self._path_exists(self.player_pos, self.exit_pos, potential=True):
                break

    def _move_player(self, movement_action):
        dx, dy = 0, 0
        if movement_action == 1: dy = -1 # Up
        elif movement_action == 2: dy = 1  # Down
        elif movement_action == 3: dx = -1 # Left
        elif movement_action == 4: dx = 1  # Right
        
        if dx != 0 or dy != 0:
            self.facing_direction = (dx, dy)
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if self._is_walkable(new_pos):
                self.player_pos = new_pos
                return True
        return False

    def _teleport_block(self):
        # Identify block to teleport
        block_to_teleport_pos = (self.player_pos[0] + self.facing_direction[0], 
                                 self.player_pos[1] + self.facing_direction[1])
        
        if not self._is_valid(block_to_teleport_pos): return 0
        
        block_type = self.grid[block_to_teleport_pos]
        if block_type not in [self.BLOCK_BLUE, self.BLOCK_GREEN]:
            return 0 # Cannot teleport this block
        
        # Identify target location
        target_pos = (block_to_teleport_pos[0] + self.facing_direction[0] * self.TELEPORT_DISTANCE,
                      block_to_teleport_pos[1] + self.facing_direction[1] * self.TELEPORT_DISTANCE)
        
        # Clamp target position to be within the grid (excluding border)
        target_pos = (max(1, min(self.grid_width-2, target_pos[0])),
                      max(1, min(self.grid_height-2, target_pos[1])))

        if self.grid[target_pos] != self.BLOCK_EMPTY:
            return 0 # Target location is not empty

        # Execute teleport
        self.grid[block_to_teleport_pos] = self.BLOCK_EMPTY
        self.grid[target_pos] = block_type
        
        self._create_particles(self._grid_to_screen(block_to_teleport_pos), self.COLOR_PLAYER, 20)
        self._create_particles(self._grid_to_screen(target_pos), self.COLOR_BLOCK_BLUE if block_type == self.BLOCK_BLUE else self.COLOR_BLOCK_GREEN, 30)

        # Handle landing effects
        landing_reward = self._handle_landing(target_pos, block_type)
        return landing_reward

    def _handle_landing(self, pos, block_type):
        if block_type == self.BLOCK_GREEN:
            return self._explode(pos)
        return 0

    def _explode(self, pos):
        self._create_particles(self._grid_to_screen(pos), self.COLOR_BLOCK_GREEN, 50, life=45, speed=4)
        self.grid[pos] = self.BLOCK_EMPTY # Green block destroys itself
        
        destruction_reward = 1.0
        
        # Use a queue for chain reactions
        explosion_queue = deque()
        
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]: # Adjacent blocks
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid(adj_pos):
                adj_block = self.grid[adj_pos]
                if adj_block in [self.BLOCK_BLUE, self.BLOCK_GREEN]:
                    if adj_block == self.BLOCK_GREEN:
                        explosion_queue.append(adj_pos) # Chain reaction
                    
                    self.grid[adj_pos] = self.BLOCK_EMPTY
                    destruction_reward += 1.0
                    self._create_particles(self._grid_to_screen(adj_pos), self.COLOR_BLOCK_BLUE, 15, speed=2)
        
        while explosion_queue:
            next_pos = explosion_queue.popleft()
            if self.grid[next_pos] != self.BLOCK_EMPTY: # Check if not already destroyed
                 destruction_reward += self._explode(next_pos)

        return destruction_reward

    def _check_termination(self):
        # 1. Win condition
        if self.player_pos == self.exit_pos:
            self.level += 1 # Progress to next level on win
            return True, 100.0

        # 2. Max steps (handled by truncation)
        if self.steps >= self.MAX_STEPS:
            return True, -10.0

        # 3. Loss condition (trapped and no way to clear a path)
        movable_blocks_exist = np.any((self.grid == self.BLOCK_BLUE) | (self.grid == self.BLOCK_GREEN))
        if not movable_blocks_exist:
            if not self._path_exists(self.player_pos, self.exit_pos, potential=False):
                return True, -10.0
                
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.grid_width + 1):
            start_pos = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y + self.grid_height * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.grid_height + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + y * self.tile_size)
            end_pos = (self.grid_offset_x + self.grid_width * self.tile_size, self.grid_offset_y + y * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw blocks and exit
        self._render_exit_portal()
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                block_type = self.grid[x, y]
                if block_type != self.BLOCK_EMPTY:
                    self._draw_block((x, y), block_type)
        
        # Draw teleport target indicator
        self._render_teleport_indicator()
        
        # Draw player
        self._render_player()

        # Draw particles
        for p in self.particles:
            p_pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], p_pos, int(p['radius']))

    def _render_player(self):
        px, py = self._grid_to_screen(self.player_pos)
        size = int(self.tile_size * 0.7)
        half_size = size // 2
        player_rect = pygame.Rect(px - half_size, py - half_size, size, size)
        
        # Glow effect
        for i in range(size // 2, 0, -2):
            alpha = 100 - (i / (size // 2)) * 100
            glow_color = (*self.COLOR_PLAYER_GLOW, alpha)
            s = pygame.Surface((size + i, size + i), pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=5)
            self.screen.blit(s, (player_rect.x - i//2, player_rect.y - i//2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_exit_portal(self):
        px, py = self._grid_to_screen(self.exit_pos)
        max_radius = self.tile_size // 2
        angle = (self.steps % 60) / 60 * 2 * math.pi
        
        for i in range(max_radius, 2, -2):
            frac = i / max_radius
            color = self.COLOR_EXIT_1 if i % 4 == 0 else self.COLOR_EXIT_2
            radius = int(max_radius * (0.5 + 0.5 * math.sin(angle + frac * math.pi)))
            if radius > 1:
                pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)

    def _draw_block(self, pos, block_type):
        px, py = self._grid_to_screen(pos)
        size = int(self.tile_size * 0.9)
        half_size = size // 2
        rect = pygame.Rect(px - half_size, py - half_size, size, size)
        
        color_map = {
            self.BLOCK_BLUE: self.COLOR_BLOCK_BLUE,
            self.BLOCK_GREEN: self.COLOR_BLOCK_GREEN,
            self.BLOCK_RED: self.COLOR_BLOCK_RED,
        }
        color = color_map.get(block_type, (255, 255, 255))
        
        pygame.draw.rect(self.screen, color, rect, border_radius=2)
        if block_type == self.BLOCK_RED: # Add detail to immovable blocks
            pygame.draw.line(self.screen, self.COLOR_BG, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, self.COLOR_BG, rect.topright, rect.bottomleft, 2)

    def _render_teleport_indicator(self):
        block_pos = (self.player_pos[0] + self.facing_direction[0], 
                     self.player_pos[1] + self.facing_direction[1])
        if not self._is_valid(block_pos) or self.grid[block_pos] not in [self.BLOCK_BLUE, self.BLOCK_GREEN]:
            return

        target_pos = (block_pos[0] + self.facing_direction[0] * self.TELEPORT_DISTANCE,
                      block_pos[1] + self.facing_direction[1] * self.TELEPORT_DISTANCE)
        target_pos = (max(1, min(self.grid_width-2, target_pos[0])),
                      max(1, min(self.grid_height-2, target_pos[1])))
        
        if self.grid[target_pos] == self.BLOCK_EMPTY:
            px, py = self._grid_to_screen(target_pos)
            size = self.tile_size
            rect = pygame.Rect(px - size//2, py - size//2, size, size)
            
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_TELEPORT_TARGET, s.get_rect(), border_radius=3)
            self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        level_text = self.font_main.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))
        
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }
        
    def _is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height
        
    def _is_walkable(self, pos):
        return self._is_valid(pos) and self.grid[pos] == self.BLOCK_EMPTY

    def _path_exists(self, start, end, potential=False):
        """Checks for a path using BFS. If potential=True, treats movable blocks as walkable."""
        q = deque([start])
        visited = {start}
        while q:
            curr = q.popleft()
            if curr == end:
                return True
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                neighbor = (curr[0] + dx, curr[1] + dy)
                if neighbor not in visited and self._is_valid(neighbor):
                    block_type = self.grid[neighbor]
                    is_passable = (block_type == self.BLOCK_EMPTY) or \
                                  (potential and block_type in [self.BLOCK_BLUE, self.BLOCK_GREEN])
                    if is_passable or neighbor == end:
                        visited.add(neighbor)
                        q.append(neighbor)
        return False
        
    def _grid_to_screen(self, pos):
        x, y = pos
        screen_x = self.grid_offset_x + x * self.tile_size + self.tile_size // 2
        screen_y = self.grid_offset_y + y * self.tile_size + self.tile_size // 2
        return int(screen_x), int(screen_y)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _create_particles(self, pos, color, count, life=30, speed=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = (math.cos(angle) * self.np_random.uniform(0.5, 1) * speed, 
                   math.sin(angle) * self.np_random.uniform(0.5, 1) * speed)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * (p['life'] / p['max_life']))
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset which calls _get_observation
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example Usage ---
    # Set the video driver to a real one for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    pygame.display.set_caption("Quantum Labyrinth")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset level")
    print("Q: Quit")
    
    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # On win, reset to the next level. On loss/timeout, reset to level 1.
            if info['player_pos'] == info['exit_pos']:
                 print(f"Level {info['level']-1} complete! Starting level {info['level']}.")
                 obs, info = env.reset(options={'level': info['level']})
            else:
                 print("Resetting to level 1.")
                 obs, info = env.reset()


        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()