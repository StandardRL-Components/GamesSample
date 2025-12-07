
# Generated: 2025-08-27T19:15:52.168229
# Source Brief: brief_02098.md
# Brief Index: 2098

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push an adjacent block. Guide the robot to the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Push blocks to clear a path for the robot to reach the exit. Plan your moves carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_DIM = 10
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_DIM * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_DIM * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.MAX_STEPS = 100
        self.NUM_BLOCKS = 25

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 65)
        self.COLOR_ROBOT = (60, 160, 255)
        self.COLOR_ROBOT_SHADOW = (40, 110, 190)
        self.COLOR_BLOCK = (255, 80, 80)
        self.COLOR_BLOCK_SHADOW = (190, 50, 50)
        self.COLOR_EXIT = (80, 255, 120)
        self.COLOR_EXIT_SHADOW = (50, 190, 80)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_UI_BG = (30, 35, 55, 180)

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Game state variables
        self.robot_pos = pygame.Vector2(0, 0)
        self.exit_pos = pygame.Vector2(0, 0)
        self.blocks = set()
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self._setup_level()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        while True:
            self.exit_pos = pygame.Vector2(self.GRID_DIM - 1, self.np_random.integers(0, self.GRID_DIM))
            self.robot_pos = pygame.Vector2(0, self.np_random.integers(0, self.GRID_DIM))
            
            self.blocks = set()
            possible_coords = [(x, y) for x in range(self.GRID_DIM) for y in range(self.GRID_DIM)]
            possible_coords.remove((self.robot_pos.x, self.robot_pos.y))
            possible_coords.remove((self.exit_pos.x, self.exit_pos.y))
            
            block_indices = self.np_random.choice(len(possible_coords), self.NUM_BLOCKS, replace=False)
            for i in block_indices:
                self.blocks.add(possible_coords[i])
            
            if not self.is_robot_stuck():
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.steps += 1
        
        push_successful = self._handle_push(movement)
        
        old_dist = self._manhattan_distance(self.robot_pos, self.exit_pos)
        self._update_robot()
        new_dist = self._manhattan_distance(self.robot_pos, self.exit_pos)
        
        if push_successful:
            reward = old_dist - new_dist
        
        terminated = False
        if self.robot_pos == self.exit_pos:
            reward = 50
            terminated = True
            self.game_over = True
            self.termination_reason = "SUCCESS: Robot reached the exit!"
        elif self.is_robot_stuck():
            reward = -10
            terminated = True
            self.game_over = True
            self.termination_reason = "FAIL: Robot is trapped!"
        elif self.steps >= self.MAX_STEPS:
            reward = -20
            terminated = True
            self.game_over = True
            self.termination_reason = f"FAIL: Exceeded {self.MAX_STEPS} moves."

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_push(self, movement_action):
        if movement_action == 0:
            return False

        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement_action]
        
        target_pos = (int(self.robot_pos.x + dx), int(self.robot_pos.y + dy))
        
        if target_pos in self.blocks:
            dest_pos = (target_pos[0] + dx, target_pos[1] + dy)
            if (0 <= dest_pos[0] < self.GRID_DIM and
                0 <= dest_pos[1] < self.GRID_DIM and
                dest_pos not in self.blocks and
                dest_pos != (int(self.exit_pos.x), int(self.exit_pos.y))):
                
                self.blocks.remove(target_pos)
                self.blocks.add(dest_pos)
                # sfx: block_push_sound
                self._create_particles(target_pos, self.COLOR_BLOCK)
                return True
        return False

    def _update_robot(self):
        path = self._find_shortest_path(self.robot_pos, self.exit_pos, self.blocks)
        if path and len(path) > 1:
            self.robot_pos = pygame.Vector2(path[1])
            # sfx: robot_move_step

    def _find_shortest_path(self, start, end, obstacles):
        start_tuple = (int(start.x), int(start.y))
        end_tuple = (int(end.x), int(end.y))

        if start_tuple == end_tuple:
            return [start_tuple]

        queue = collections.deque([(start_tuple, [start_tuple])])
        visited = {start_tuple}

        while queue:
            current_pos, path = queue.popleft()
            if current_pos == end_tuple:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if (0 <= next_pos[0] < self.GRID_DIM and
                    0 <= next_pos[1] < self.GRID_DIM and
                    next_pos not in obstacles and
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    new_path = list(path)
                    new_path.append(next_pos)
                    queue.append((next_pos, new_path))
        return None

    def is_robot_stuck(self):
        return self._find_shortest_path(self.robot_pos, self.exit_pos, self.blocks) is None

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y), 
                             (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT))
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE), 
                             (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE))

        # Draw exit
        self._draw_grid_object(self.exit_pos, self.COLOR_EXIT, self.COLOR_EXIT_SHADOW)
        
        # Draw blocks
        for block_pos in self.blocks:
            self._draw_grid_object(pygame.Vector2(block_pos), self.COLOR_BLOCK, self.COLOR_BLOCK_SHADOW)
            
        # Draw robot
        self._draw_grid_object(self.robot_pos, self.COLOR_ROBOT, self.COLOR_ROBOT_SHADOW)

        # Update and draw particles
        self._update_and_draw_particles()

    def _draw_grid_object(self, pos, color, shadow_color):
        x = self.GRID_OFFSET_X + pos.x * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + pos.y * self.CELL_SIZE
        shadow_offset = 3
        
        main_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw shadow rects for a 3D effect
        pygame.draw.rect(self.screen, shadow_color, main_rect)
        pygame.draw.rect(self.screen, color, (x, y, self.CELL_SIZE - shadow_offset, self.CELL_SIZE - shadow_offset))

    def _render_ui(self):
        # Moves display
        moves_text = self.font_main.render(f"Moves: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topleft=(10, 10))
        ui_box = moves_rect.inflate(10, 5)
        pygame.gfxdraw.box(self.screen, ui_box, self.COLOR_UI_BG)
        self.screen.blit(moves_text, moves_rect)
        
        # Score display
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        ui_box = score_rect.inflate(10, 5)
        pygame.gfxdraw.box(self.screen, ui_box, self.COLOR_UI_BG)
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_main.render(self.termination_reason, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _create_particles(self, grid_pos, color):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = self.np_random.uniform(3, 7)
            lifetime = self.np_random.uniform(10, 20)
            self.particles.append([pygame.Vector2(px, py), vel, size, lifetime, color])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0] += p[1]  # Update position
            p[2] -= 0.2   # Shrink size
            p[3] -= 1     # Reduce lifetime
            if p[3] <= 0 or p[2] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p[3] / 20))))
                color_with_alpha = (*p[4], alpha)
                temp_surf = pygame.Surface((p[2]*2, p[2]*2), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color_with_alpha, temp_surf.get_rect())
                self.screen.blit(temp_surf, p[0] - pygame.Vector2(p[2], p[2]))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot_pos": (self.robot_pos.x, self.robot_pos.y),
            "exit_pos": (self.exit_pos.x, self.exit_pos.y),
            "is_stuck": self.is_robot_stuck(),
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Pusher Bot")
    clock = pygame.time.Clock()
    
    running = True
    game_over = False
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_over:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                if event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    game_over = False
                    print("--- Game Reset ---")

        if action[0] != 0: # Only step if a move key was pressed
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated or truncated
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()