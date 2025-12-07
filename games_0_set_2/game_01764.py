
# Generated: 2025-08-27T18:12:42.362444
# Source Brief: brief_01764.md
# Brief Index: 1764

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your robot one tile at a time. Reach the green goal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through obstacle-laden isometric arenas to reach the goal as fast as possible."
    )

    # Frames auto-advance for smooth graphics and time-based challenges.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 64)
        self.COLOR_ROBOT = (52, 152, 219)
        self.COLOR_ROBOT_GLOW = (52, 152, 219, 50)
        self.COLOR_GOAL = (46, 204, 113)
        self.COLOR_GOAL_GLOW = (46, 204, 113, 60)
        self.COLOR_OBSTACLE = (231, 76, 60)
        self.COLOR_OBSTACLE_GLOW = (231, 76, 60, 60)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_TIMER_BAR = (241, 196, 15)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 48)

        # --- Game State ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.robot_grid_pos = (0, 0)
        self.robot_vis_pos = (0, 0)
        self.is_moving = False
        self.move_cooldown = 0
        self.goal_pos = (0, 0)
        self.obstacles = []
        self.moving_obstacles = []
        self.particles = []
        self.time_remaining = 0
        self.INITIAL_TIME_PER_STAGE = 10 * self.FPS
        self.max_steps = 3 * self.INITIAL_TIME_PER_STAGE + 100 # Max time for 3 stages + buffer

        # --- Isometric Projection ---
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        self._define_stages()
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def _define_stages(self):
        self.stage_data = {
            1: {
                "grid_size": (10, 10),
                "start": (1, 1),
                "goal": (8, 8),
                "obstacles": [(3, 3), (3, 4), (4, 3), (6, 6), (6, 7), (7, 6)],
                "moving_obstacles": []
            },
            2: {
                "grid_size": (12, 12),
                "start": (1, 1),
                "goal": (10, 10),
                "obstacles": [(3, 1), (3, 2), (3, 3), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5),
                              (7, 10), (7, 9), (7, 8), (7, 7), (9, 3), (9, 4), (9, 5), (10, 3)],
                "moving_obstacles": []
            },
            3: {
                "grid_size": (15, 15),
                "start": (1, 1),
                "goal": (13, 13),
                "obstacles": [(3, 3), (4, 3), (5, 3), (3, 4), (3, 5), (7, 0), (7, 1), (7, 2),
                              (0, 8), (1, 8), (2, 8), (3, 8), (10, 10), (11, 10), (10, 11),
                              (13, 5), (13, 6), (13, 7)],
                "moving_obstacles": [
                    {'path': [(5, 8), (10, 8)], 'speed': 0.03},
                    {'path': [(8, 5), (8, 12)], 'speed': 0.04},
                    {'path': [(1, 12), (5, 12)], 'speed': 0.02},
                ]
            }
        }

    def _load_stage(self, stage_num):
        data = self.stage_data[stage_num]
        self.grid_size = data["grid_size"]
        self.robot_grid_pos = data["start"]
        self.robot_vis_pos = self._grid_to_iso(*self.robot_grid_pos)
        self.goal_pos = data["goal"]
        self.obstacles = [tuple(o) for o in data["obstacles"]]
        self.moving_obstacles = []
        for mo in data["moving_obstacles"]:
            self.moving_obstacles.append({
                'path': mo['path'],
                'pos': list(mo['path'][0]),
                'target_idx': 1,
                'speed': mo['speed']
            })
        self.time_remaining = self.INITIAL_TIME_PER_STAGE
        self.is_moving = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.particles = []
        self._load_stage(self.current_stage)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Processing ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = -0.01  # Small penalty for time passing
        terminated = False

        self.steps += 1
        self.time_remaining -= 1
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        # --- Update Game Logic ---
        self._update_robot_movement(movement)
        self._update_moving_obstacles()
        self._update_particles()

        # --- Collision and Goal Checks ---
        collision, goal_reached = self._check_events()

        if collision:
            reward = -10
            self.score -= 10
            terminated = True
            self.game_over = True
            # SFX: Explosion
        
        if goal_reached:
            time_bonus = 10 * (self.time_remaining / self.INITIAL_TIME_PER_STAGE)
            stage_reward = 5 + time_bonus
            reward += stage_reward
            self.score += stage_reward
            # SFX: Goal reached chime

            if self.current_stage < 3:
                self.current_stage += 1
                self._load_stage(self.current_stage)
            else:
                win_bonus = 50
                reward += win_bonus
                self.score += win_bonus
                terminated = True
                self.game_over = True
        
        if self.time_remaining <= 0 or self.steps >= self.max_steps:
            if not terminated: # Avoid double penalty if goal reached on last frame
                reward = -10
                self.score -= 10
            terminated = True
            self.game_over = True
            # SFX: Timeout buzzer

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_robot_movement(self, movement_action):
        # Interpolate visual position for smooth animation
        target_vis_pos = self._grid_to_iso(*self.robot_grid_pos)
        dx = target_vis_pos[0] - self.robot_vis_pos[0]
        dy = target_vis_pos[1] - self.robot_vis_pos[1]
        self.robot_vis_pos = (self.robot_vis_pos[0] + dx * 0.3, self.robot_vis_pos[1] + dy * 0.3)

        if math.hypot(dx, dy) < 1.0:
            self.robot_vis_pos = target_vis_pos
            self.is_moving = False

        if not self.is_moving and self.move_cooldown == 0 and movement_action != 0:
            gx, gy = self.robot_grid_pos
            next_pos = gx, gy
            if movement_action == 1: next_pos = (gx, gy - 1) # Up
            elif movement_action == 2: next_pos = (gx, gy + 1) # Down
            elif movement_action == 3: next_pos = (gx - 1, gy) # Left
            elif movement_action == 4: next_pos = (gx + 1, gy) # Right

            # Check bounds
            if 0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1]:
                self.robot_grid_pos = next_pos
                self.is_moving = True
                self.move_cooldown = 5 # Cooldown in frames to prevent instant multi-moves
                # SFX: Robot move whoosh
                self._spawn_particles(self.robot_vis_pos, self.COLOR_ROBOT, 10)

    def _update_moving_obstacles(self):
        for mo in self.moving_obstacles:
            target_pos = mo['path'][mo['target_idx']]
            current_pos = mo['pos']
            
            direction = [target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]]
            dist = math.hypot(*direction)
            
            if dist < 0.1:
                mo['target_idx'] = 1 - mo['target_idx'] # Flip between 0 and 1
            else:
                norm_dir = [d / dist for d in direction]
                mo['pos'][0] += norm_dir[0] * mo['speed'] * self.TILE_WIDTH * 0.5
                mo['pos'][1] += norm_dir[1] * mo['speed'] * self.TILE_WIDTH * 0.5

    def _check_events(self):
        collision = False
        goal_reached = False

        # Check static obstacles
        if self.robot_grid_pos in self.obstacles:
            collision = True

        # Check moving obstacles
        for mo in self.moving_obstacles:
            mo_grid_pos = (round(mo['pos'][0]), round(mo['pos'][1]))
            if self.robot_grid_pos == mo_grid_pos:
                collision = True
                break
        
        # Check goal
        if self.robot_grid_pos == self.goal_pos:
            goal_reached = True
            
        return collision, goal_reached

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for r in range(self.grid_size[1]):
            for c in range(self.grid_size[0]):
                self._draw_iso_tile((c, r), self.COLOR_GRID, filled=False)
        
        # Render goal
        self._draw_iso_tile(self.goal_pos, self.COLOR_GOAL_GLOW, size_mod=6)
        self._draw_iso_tile(self.goal_pos, self.COLOR_GOAL)

        # Render static obstacles
        for obs_pos in self.obstacles:
            self._draw_iso_tile(obs_pos, self.COLOR_OBSTACLE_GLOW, size_mod=6)
            self._draw_iso_tile(obs_pos, self.COLOR_OBSTACLE)

        # Render moving obstacles
        for mo in self.moving_obstacles:
            pos_tuple = (mo['pos'][0], mo['pos'][1])
            self._draw_iso_tile(pos_tuple, self.COLOR_OBSTACLE_GLOW, size_mod=6)
            self._draw_iso_tile(pos_tuple, self.COLOR_OBSTACLE)

        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

        # Render robot
        self._draw_iso_tile(self.robot_vis_pos, self.COLOR_ROBOT_GLOW, is_vis_pos=True, size_mod=8)
        self._draw_iso_tile(self.robot_vis_pos, self.COLOR_ROBOT, is_vis_pos=True)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.current_stage}/3", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.WIDTH // 2, y=10)
        self.screen.blit(stage_text, stage_rect)

        # Time bar
        timer_width = 150
        timer_height = 15
        time_pct = max(0, self.time_remaining / self.INITIAL_TIME_PER_STAGE)
        bar_x = self.WIDTH - timer_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, timer_width, timer_height))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (bar_x, bar_y, timer_width * time_pct, timer_height))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.current_stage == 3 and self.robot_grid_pos == self.goal_pos:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
                
            end_text = self.font_big.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_remaining": self.time_remaining
        }

    # --- Helper & Rendering Functions ---
    def _grid_to_iso(self, gx, gy):
        sx = self.ORIGIN_X + (gx - gy) * (self.TILE_WIDTH / 2)
        sy = self.ORIGIN_Y + (gx + gy) * (self.TILE_HEIGHT / 2)
        return sx, sy

    def _draw_iso_tile(self, pos, color, is_vis_pos=False, filled=True, size_mod=0):
        if is_vis_pos:
            sx, sy = pos
        else:
            sx, sy = self._grid_to_iso(pos[0], pos[1])
        
        w = self.TILE_WIDTH + size_mod
        h = self.TILE_HEIGHT + size_mod
        
        points = [
            (sx, sy - h / 2),
            (sx + w / 2, sy),
            (sx, sy + h / 2),
            (sx - w / 2, sy)
        ]
        
        int_points = [(int(p[0]), int(p[1])) for p in points]

        if filled:
            pygame.gfxdraw.filled_polygon(self.screen, int_points, color)
        else:
            pygame.gfxdraw.aapolygon(self.screen, int_points, color)
            
    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'x': pos[0],
                'y': pos[1],
                'vx': self.np_random.uniform(-1, 1),
                'vy': self.np_random.uniform(-1, 1),
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['size'] -= 0.1
            p['life'] -= 1
            if p['size'] <= 0 or p['life'] <= 0:
                self.particles.remove(p)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# This block allows you to run the game directly for testing
if __name__ == '__main__':
    import os
    # Set a specific video driver to run pygame headlessly if needed,
    # but for direct play, we want to see the window.
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 

    env = GameEnv()
    
    # --- Human Player Controls ---
    # This setup is for demonstration and debugging.
    # An RL agent would call env.step(action) directly.
    
    obs, info = env.reset()
    done = False
    
    # Override the screen to be a display surface
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Robot Arena")

    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
        # Map keyboard to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([movement, space, shift])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering for Human Player ---
        # The observation is already the rendered frame, so we just need
        # to get it back onto the display surface.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    print(f"Game Over. Final Score: {info['score']}")
    
    # Wait a bit before closing
    pygame.time.wait(2000)
    pygame.quit()