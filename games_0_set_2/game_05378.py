
# Generated: 2025-08-28T04:49:54.159763
# Source Brief: brief_05378.md
# Brief Index: 5378

        
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
        "Controls: Arrow keys to move the selected robot. No-Op (no key press) to cycle to the next robot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide your squad of 5 robots to their matching charging stations before time or moves run out. Plan your path efficiently in this isometric puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()

        # Game constants
        self.FPS = 30
        self.NUM_ROBOTS = 5
        self.MAX_TIME = 180  # seconds
        self.MAX_MOVES = 500
        self.GRID_W, self.GRID_H = 16, 11
        self.TILE_W, self.TILE_H = 44, 22
        self.ORIGIN_X, self.ORIGIN_Y = self.screen_size[0] // 2, 100

        # Visuals
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (35, 35, 45)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_OBSTACLE = (100, 100, 110)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_HIGHLIGHT = (255, 255, 255)
        self.ROBOT_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (80, 255, 255),  # Cyan
        ]

        # State variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.time_remaining = 0
        self.moves_made = 0
        self.selected_robot_idx = 0
        self.robot_grid_pos = []
        self.robot_visual_pos = []
        self.robot_is_moving = []
        self.charger_grid_pos = []
        self.obstacle_grid_pos = []
        self.is_charged = []
        
        self.reset()
        
        self.validate_implementation()

    def _to_iso(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_W / 2
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_H / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, color, grid_x, grid_y, z_offset=0, scale=1.0):
        sx, sy = self._to_iso(grid_x, grid_y)
        sy -= z_offset
        
        scaled_w = self.TILE_W * scale
        scaled_h = self.TILE_H * scale

        points = [
            (sx, sy - scaled_h / 2),
            (sx + scaled_w / 2, sy),
            (sx, sy + scaled_h / 2),
            (sx - scaled_w / 2, sy),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.time_remaining = self.MAX_TIME
        self.moves_made = 0
        self.selected_robot_idx = 0
        
        # Generate a solvable puzzle
        all_pos = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_pos)
        
        self.charger_grid_pos = [all_pos.pop() for _ in range(self.NUM_ROBOTS)]
        self.robot_grid_pos = [all_pos.pop() for _ in range(self.NUM_ROBOTS)]
        self.obstacle_grid_pos = [] # No obstacles in this version

        self.robot_visual_pos = [list(pos) for pos in self.robot_grid_pos]
        self.robot_is_moving = [False] * self.NUM_ROBOTS
        self.is_charged = [False] * self.NUM_ROBOTS
        self._update_charged_status()
        
        return self._get_observation(), self._get_info()

    def _is_valid_move(self, target_pos, moving_robot_idx):
        # Check bounds
        if not (0 <= target_pos[0] < self.GRID_W and 0 <= target_pos[1] < self.GRID_H):
            return False
        # Check obstacles
        if target_pos in self.obstacle_grid_pos:
            return False
        # Check other robots
        for i in range(self.NUM_ROBOTS):
            if i != moving_robot_idx and target_pos == self.robot_grid_pos[i]:
                return False
        return True

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _update_animations(self):
        for i in range(self.NUM_ROBOTS):
            if self.robot_is_moving[i]:
                # Lerp visual position towards logical grid position
                vx, vy = self.robot_visual_pos[i]
                tx, ty = self.robot_grid_pos[i]
                
                self.robot_visual_pos[i][0] = vx + (tx - vx) * 0.2
                self.robot_visual_pos[i][1] = vy + (ty - vy) * 0.2

                # Stop moving when close enough
                if abs(tx - self.robot_visual_pos[i][0]) < 0.01 and abs(ty - self.robot_visual_pos[i][1]) < 0.01:
                    self.robot_visual_pos[i] = list(self.robot_grid_pos[i])
                    self.robot_is_moving[i] = False
    
    def _update_charged_status(self):
        for i in range(self.NUM_ROBOTS):
            self.is_charged[i] = (self.robot_grid_pos[i] == self.charger_grid_pos[i])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        reward = 0

        self._update_animations()

        movement = action[0]
        
        # Action: Cycle selected robot
        if movement == 0:
            if not any(self.robot_is_moving):
                self.selected_robot_idx = (self.selected_robot_idx + 1) % self.NUM_ROBOTS
        
        # Action: Move robot
        elif movement in [1, 2, 3, 4]:
            idx = self.selected_robot_idx
            if not self.robot_is_moving[idx] and not self.is_charged[idx] and self.moves_made < self.MAX_MOVES:
                dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1] # up, down, left, right
                curr_pos = self.robot_grid_pos[idx]
                target_pos = (curr_pos[0] + dx, curr_pos[1] + dy)
                
                if self._is_valid_move(target_pos, idx):
                    # Sound: sfx_robot_move.wav
                    self.moves_made += 1
                    
                    old_dist = self._manhattan_distance(curr_pos, self.charger_grid_pos[idx])
                    new_dist = self._manhattan_distance(target_pos, self.charger_grid_pos[idx])
                    
                    if new_dist < old_dist: reward += 0.1
                    elif new_dist > old_dist: reward -= 0.1
                    
                    self.robot_grid_pos[idx] = target_pos
                    self.robot_is_moving[idx] = True

                    was_charged = self.is_charged[idx]
                    self._update_charged_status()
                    is_now_charged = self.is_charged[idx]
                    
                    if is_now_charged and not was_charged:
                        # Sound: sfx_charge_up.wav
                        reward += 10
        
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            if all(self.is_charged):
                reward += 100
                self.win_message = "SUCCESS"
            else:
                reward -= 100
                if self.time_remaining <= 0: self.win_message = "TIME OUT"
                else: self.win_message = "MOVE LIMIT"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.game_over: return True
        
        if all(self.is_charged):
            self.game_over = True
            return True
        if self.time_remaining <= 0 or self.moves_made >= self.MAX_MOVES:
            self.game_over = True
            return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                self._draw_iso_poly(self.screen, self.COLOR_GRID, x, y, z_offset=-1, scale=0.95)

        # Draw chargers
        for i in range(self.NUM_ROBOTS):
            gx, gy = self.charger_grid_pos[i]
            color = self.ROBOT_COLORS[i]
            
            # Glow effect
            glow_color = color[:3] + (30 + int(20 * math.sin(self.steps * 0.1)),)
            self._draw_iso_poly(self.screen, glow_color, gx, gy, z_offset=0, scale=1.2)
            
            base_color = tuple(c // 2 for c in color)
            self._draw_iso_poly(self.screen, base_color, gx, gy, z_offset=0, scale=0.8)

        # Draw robots
        for i in range(self.NUM_ROBOTS):
            vx, vy = self.robot_visual_pos[i]
            color = self.ROBOT_COLORS[i]
            
            z = 0
            scale = 0.7
            if self.is_charged[i]:
                z = -self.TILE_H * 0.2
                scale = 0.6
                
            # Shadow
            shadow_sx, shadow_sy = self._to_iso(vx, vy)
            pygame.gfxdraw.filled_ellipse(self.screen, shadow_sx, shadow_sy + int(self.TILE_H * 0.4), int(self.TILE_W*0.3), int(self.TILE_H*0.15), (0,0,0,100))
            
            # Robot body
            self._draw_iso_poly(self.screen, color, vx, vy, z_offset=self.TILE_H * 0.5 + z, scale=scale)

            # Highlight selected robot
            if i == self.selected_robot_idx and not self.is_charged[i] and not self.game_over:
                pulse = 0.8 + 0.2 * math.sin(self.steps * 0.2)
                highlight_color = self.COLOR_HIGHLIGHT[:3] + (int(150 * pulse),)
                self._draw_iso_poly(self.screen, highlight_color, vx, vy, z_offset=self.TILE_H * 0.5 + z, scale=scale * 1.2)

    def _render_ui(self):
        # Moves counter
        moves_text = self.font_m.render(f"MOVES: {self.moves_made}/{self.MAX_MOVES}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))

        # Time counter
        time_text = self.font_m.render(f"TIME: {max(0, int(self.time_remaining))}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.screen_size[0] - 20, 15))
        self.screen.blit(time_text, time_rect)

        # Charged status
        icon_size = 20
        total_width = self.NUM_ROBOTS * (icon_size + 10) - 10
        start_x = (self.screen_size[0] - total_width) // 2
        for i in range(self.NUM_ROBOTS):
            x = start_x + i * (icon_size + 10)
            y = self.screen_size[1] - 30
            rect = pygame.Rect(x, y, icon_size, icon_size)
            
            if self.is_charged[i]:
                pygame.draw.rect(self.screen, self.ROBOT_COLORS[i], rect, border_radius=4)
            else:
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, width=2, border_radius=4)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_l.render(self.win_message, True, self.COLOR_HIGHLIGHT)
            end_rect = end_text.get_rect(center=(self.screen_size[0] / 2, self.screen_size[1] / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": round(self.time_remaining, 2),
            "moves_made": self.moves_made,
            "robots_charged": sum(1 for c in self.is_charged if c),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Robot Charger")
    
    terminated = False
    action = np.array([0, 0, 0]) # Start with no-op

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
        
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op default
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space, shift])
        # --- End action mapping ---

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()