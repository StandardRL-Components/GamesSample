
# Generated: 2025-08-28T04:33:02.234609
# Source Brief: brief_05286.md
# Brief Index: 5286

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Collect all the fruit to advance to the next stage."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Navigate the grid to collect all the fruit while avoiding the ghosts."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 30
    GRID_HEIGHT = 18
    CELL_SIZE = 20
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2  # 20
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) + 10 # 30

    # Colors
    COLOR_BG = (0, 0, 0)
    COLOR_GRID = (40, 40, 40)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_FRUIT = (255, 0, 0)
    COLOR_FRUIT_HIGHLIGHT = (255, 150, 150)
    GHOST_COLORS = [(0, 255, 255), (255, 0, 255), (255, 184, 82)] # Cyan, Magenta, Orange
    COLOR_TEXT = (255, 255, 255)
    COLOR_EYE = (255, 255, 255)
    COLOR_PUPIL = (0, 0, 0)

    # Game parameters
    MAX_STEPS = 1000
    TOTAL_STAGES = 3
    FRUITS_PER_STAGE = 20
    GHOST_MOVE_INTERVAL_BASE = 20
    
    # Rewards
    REWARD_COLLECT_FRUIT = 10
    REWARD_STAGE_CLEAR = 100
    REWARD_GAME_WIN = 300
    REWARD_CLOSER_TO_FRUIT = 1.0
    REWARD_CLOSER_TO_GHOST = -0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_stage = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game state variables are initialized in reset()
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_dir = (1, 0) # For rendering direction
        self.player_anim_state = 0
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the current stage with fruits and ghosts."""
        # Set ghost speed (lower interval is faster)
        self.ghost_move_interval = max(5, self.GHOST_MOVE_INTERVAL_BASE - (self.stage - 1))

        # Place fruits
        self.fruits = []
        occupied_positions = {self.player_pos}
        while len(self.fruits) < self.FRUITS_PER_STAGE:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT),
            )
            if pos not in occupied_positions:
                self.fruits.append(pos)
                occupied_positions.add(pos)

        # Place ghosts
        self.ghosts = []
        ghost_starts = [(1, 1), (self.GRID_WIDTH - 2, 1), (1, self.GRID_HEIGHT - 2)]
        for i in range(min(self.stage, len(self.GHOST_COLORS))):
            start_pos = ghost_starts[i]
            if start_pos in occupied_positions: # Ensure ghost doesn't start on a fruit
                start_pos = (start_pos[0]+1, start_pos[1])
            occupied_positions.add(start_pos)

            self.ghosts.append({
                "pos": start_pos,
                "color": self.GHOST_COLORS[i],
                "path_index": 0,
                "path": self._generate_ghost_path(start_pos)
            })

    def _generate_ghost_path(self, start_pos):
        """Generates a simple patrol path for a ghost."""
        path_type = self.np_random.integers(0, 3)
        path = [start_pos]
        cx, cy = start_pos
        if path_type == 0: # Square
            w, h = self.np_random.integers(4, 8), self.np_random.integers(4, 8)
            for _ in range(w): path.append((path[-1][0] + 1, path[-1][1]))
            for _ in range(h): path.append((path[-1][0], path[-1][1] + 1))
            for _ in range(w): path.append((path[-1][0] - 1, path[-1][1]))
            for _ in range(h): path.append((path[-1][0], path[-1][1] - 1))
        elif path_type == 1: # Horizontal line
            length = self.np_random.integers(5, 12)
            for i in range(1, length): path.append((cx + i, cy))
            for i in range(length - 1, -1, -1): path.append((cx + i, cy))
        else: # Vertical line
            length = self.np_random.integers(5, 10)
            for i in range(1, length): path.append((cx, cy + i))
            for i in range(length - 1, -1, -1): path.append((cx, cy + i))
        
        # Clamp path to be within grid bounds
        return [(max(0, min(self.GRID_WIDTH - 1, x)), max(0, min(self.GRID_HEIGHT - 1, y))) for x, y in path]


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0
        
        # --- Proximity Reward Calculation (Before Move) ---
        old_dist_fruit = self._get_dist_to_nearest(self.player_pos, self.fruits)
        ghost_positions = [g['pos'] for g in self.ghosts] if self.ghosts else []
        old_dist_ghost = self._get_dist_to_nearest(self.player_pos, ghost_positions)

        # --- Player Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        if dx != 0 or dy != 0:
            self.player_dir = (dx, dy)
        
        next_px = self.player_pos[0] + dx
        next_py = self.player_pos[1] + dy

        if 0 <= next_px < self.GRID_WIDTH and 0 <= next_py < self.GRID_HEIGHT:
            self.player_pos = (next_px, next_py)
        
        self.player_anim_state = (self.steps // 4) % 2 # Animate mouth every 4 steps

        # --- Proximity Reward Calculation (After Move) ---
        new_dist_fruit = self._get_dist_to_nearest(self.player_pos, self.fruits)
        new_dist_ghost = self._get_dist_to_nearest(self.player_pos, ghost_positions)

        if new_dist_fruit < old_dist_fruit:
            reward += self.REWARD_CLOSER_TO_FRUIT
        if new_dist_ghost < old_dist_ghost:
            reward += self.REWARD_CLOSER_TO_GHOST
        
        # --- Ghost Movement ---
        if self.ghosts and self.steps % self.ghost_move_interval == 0:
            for ghost in self.ghosts:
                # # Sound: ghost_move.wav
                if ghost["path"]:
                    ghost['path_index'] = (ghost['path_index'] + 1) % len(ghost['path'])
                    ghost['pos'] = ghost['path'][ghost['path_index']]

        # --- Collision Detection ---
        terminated = False
        # Player-Ghost collision
        for ghost in self.ghosts:
            if self.player_pos == ghost['pos']:
                # # Sound: player_death.wav
                self.game_over = True
                terminated = True
                break
        
        # Player-Fruit collection
        if self.player_pos in self.fruits:
            # # Sound: collect_fruit.wav
            self.fruits.remove(self.player_pos)
            self.score += self.REWARD_COLLECT_FRUIT
            reward += self.REWARD_COLLECT_FRUIT

        # --- Stage/Game End Check ---
        if not self.fruits: # Stage cleared
            self.score += self.REWARD_STAGE_CLEAR
            reward += self.REWARD_STAGE_CLEAR
            if self.stage == self.TOTAL_STAGES:
                # # Sound: game_win.wav
                self.game_over = True
                terminated = True
                self.score += self.REWARD_GAME_WIN
                reward += self.REWARD_GAME_WIN
            else:
                # # Sound: stage_clear.wav
                self.stage += 1
                self._setup_stage()
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest(self, pos, targets):
        """Calculates Manhattan distance to the nearest target."""
        if not targets:
            return 0 # No targets, no distance to improve
        px, py = pos
        min_dist = float('inf')
        for tx, ty in targets:
            dist = abs(px - tx) + abs(py - ty)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "fruits_left": len(self.fruits)
        }

    def _render_game(self):
        self._draw_grid()
        
        for fruit_pos in self.fruits:
            self._draw_fruit(fruit_pos)
            
        for ghost in self.ghosts:
            self._draw_ghost(ghost)
            
        self._draw_player()

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, py))

    def _grid_to_pixel(self, grid_pos):
        gx, gy = grid_pos
        px = self.GRID_OFFSET_X + gx * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _draw_player(self):
        px, py = self._grid_to_pixel(self.player_pos)
        radius = self.CELL_SIZE // 2 - 2
        
        angle_offset = 0
        if self.player_dir == (1, 0): angle_offset = 0
        elif self.player_dir == (-1, 0): angle_offset = 180
        elif self.player_dir == (0, -1): angle_offset = 90
        elif self.player_dir == (0, 1): angle_offset = 270

        mouth_angle = 40 if self.player_anim_state == 1 and not self.game_over else 5
        
        start_angle_rad = math.radians(angle_offset + mouth_angle)
        end_angle_rad = math.radians(angle_offset + 360 - mouth_angle)
        
        points = [(px, py)]
        num_segments = 50
        for i in range(num_segments + 1):
            angle = start_angle_rad + (end_angle_rad - start_angle_rad) * i / num_segments
            points.append((px + radius * math.cos(angle), py + radius * math.sin(angle)))
        
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_fruit(self, pos):
        px, py = self._grid_to_pixel(pos)
        radius = self.CELL_SIZE // 4
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_FRUIT)
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_FRUIT)
        pygame.gfxdraw.aacircle(self.screen, px-1, py-1, radius//2, self.COLOR_FRUIT_HIGHLIGHT)
        pygame.gfxdraw.filled_circle(self.screen, px-1, py-1, radius//2, self.COLOR_FRUIT_HIGHLIGHT)
        
    def _draw_ghost(self, ghost):
        px, py = self._grid_to_pixel(ghost['pos'])
        radius = self.CELL_SIZE // 2 - 3
        
        bob = math.sin(self.steps * 0.2 + ghost['color'][0]) * 2
        py += int(bob)
        
        body_rect = pygame.Rect(px - radius, py, radius * 2, radius)
        pygame.draw.arc(self.screen, ghost['color'], (px - radius, py - radius, radius * 2, radius * 2), 0, math.pi, radius)
        pygame.draw.rect(self.screen, ghost['color'], body_rect)

        for i in range(3):
            sp_x = px - radius + i * (radius)
            pygame.draw.polygon(self.screen, ghost['color'], [(sp_x, py + radius), (sp_x + radius, py + radius), (sp_x + radius/2, py + radius + 5)])

        eye_radius = max(1, radius // 3)
        pupil_radius = max(1, eye_radius // 2)
        eye_offset_x = radius // 2
        
        pupil_dx = self.player_dir[0] * 2 if not self.game_over else 0
        pupil_dy = self.player_dir[1] * 2 if not self.game_over else 0
        
        for i in [-1, 1]:
            ex, ey = px + i * eye_offset_x, py - 1
            pygame.gfxdraw.aacircle(self.screen, ex, ey, eye_radius, self.COLOR_EYE)
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, eye_radius, self.COLOR_EYE)
            pygame.gfxdraw.filled_circle(self.screen, ex + pupil_dx, ey + pupil_dy, pupil_radius, self.COLOR_PUPIL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Fruits left
        fruit_text = self.font_ui.render(f"FRUIT: {len(self.fruits)}", True, self.COLOR_TEXT)
        text_rect = fruit_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(fruit_text, text_rect)
        
        # Stage
        stage_text_str = f"STAGE {self.stage}"
        if self.game_over:
            if not self.fruits and self.stage == self.TOTAL_STAGES:
                 stage_text_str = "YOU WIN!"
            else:
                 stage_text_str = "GAME OVER"
                 
        stage_text = self.font_stage.render(stage_text_str, True, self.COLOR_TEXT)
        text_rect = stage_text.get_rect(center=(self.SCREEN_WIDTH // 2, 25))
        self.screen.blit(stage_text, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Grid Collector")
        
        terminated = False
        action = env.action_space.sample()
        action[0] = 0 # Start with no-op

        while True:
            current_action = 0 # Default to no-op
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
            
            if not terminated:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: current_action = 1
                elif keys[pygame.K_DOWN]: current_action = 2
                elif keys[pygame.K_LEFT]: current_action = 3
                elif keys[pygame.K_RIGHT]: current_action = 4
            
            action[0] = current_action
            obs, reward, terminated, truncated, info = env.step(action)
            
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}. Press 'R' to restart.")

            env.clock.tick(10) # Limit FPS for human play

    except pygame.error as e:
        print("Pygame display error. This is expected in a headless environment.")
        print("Running a simple step test instead.")
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished after {_ + 1} steps. Score: {info['score']}")
                obs, info = env.reset()
    finally:
        pygame.quit()