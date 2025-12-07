
# Generated: 2025-08-28T05:01:30.639838
# Source Brief: brief_02493.md
# Brief Index: 2493

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ←→ to aim, ↑↓ to adjust power. Press Space to putt."
    )

    game_description = (
        "A minimalist isometric golf game. Sink the ball in all 9 holes with the fewest strokes. "
        "Watch out for slopes and out-of-bounds areas."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_GREEN = [(80, 160, 90), (90, 170, 100), (100, 180, 110), (110, 190, 120)]
        self.COLOR_ROUGH = (60, 120, 70)
        self.COLOR_OOB = (192, 20, 30)
        self.COLOR_HOLE = (0, 0, 0)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 100)
        self.COLOR_FLAG = (220, 200, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_AIM_LINE = (255, 255, 255, 150)
        self.COLOR_POWER_BAR_BG = (50, 50, 50)
        self.COLOR_POWER_BAR_FILL = (255, 80, 80)

        # Game constants
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.MAX_POWER = 15.0
        self.POWER_INCREMENT = 0.2
        self.AIM_INCREMENT = 0.05  # Radians
        self.FRICTION = 0.97
        self.SLOPE_FORCE_MULTIPLIER = 0.04
        self.STOP_VELOCITY_THRESHOLD = 0.05
        self.BALL_RADIUS = 5
        self.HOLE_RADIUS = 7
        self.MAX_EPISODE_STEPS = 1000
        self.PAR_PER_HOLE = 3
        self.TOTAL_PAR = self.PAR_PER_HOLE * 9
        self.MAX_STROKES = self.TOTAL_PAR * 2
        
        # Hole Definitions
        self._hole_definitions = self._create_hole_definitions()
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_strokes = 0
        self.current_hole_index = 0
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.last_ball_pos = np.array([0.0, 0.0])
        self.game_state = "AIMING"  # AIMING, BALL_MOVING, HOLE_COMPLETE
        self.aim_angle = 0.0
        self.putt_power = self.MAX_POWER / 2
        self.prev_space_held = False
        self.camera_offset = np.array([0.0, 0.0])
        self.current_hole_layout = None
        self.current_hole_height = None
        self.current_tee_pos = None
        self.current_hole_pos = None
        self.max_hole_dist = 1.0

        # Initialize state
        self.reset()
        
        # Validate
        self.validate_implementation()

    def _create_hole_definitions(self):
        defs = []
        # Hole 1: Straight shot
        defs.append({
            "layout": [
                "OOOOOOOOOOOO",
                "O          O",
                "O T      H O",
                "O          O",
                "OOOOOOOOOOOO",
            ],
            "height": [[1]*12 for _ in range(5)]
        })
        # Hole 2: Simple slope
        defs.append({
            "layout": [
                "OOOOOOOOOOOOOO",
                "O T          O",
                "O            O",
                "O          H O",
                "OOOOOOOOOOOOOO",
            ],
            "height": [
                [1]*14, [1]*14, [2]*14, [3]*14, [3]*14
            ]
        })
        # Hole 3: Dogleg right with rough
        defs.append({
            "layout": [
                "OOOOOOOOO",
                "ORRRRRRRO",
                "O T R   O",
                "O   R   O",
                "O   R H O",
                "O   RRRRO",
                "OOOOOOOOO",
            ],
            "height": [
                [1]*9, [1]*9, [1,1,1,2,2,2,1,1,1], [1,1,1,2,1,1,1,1,1],
                [1,1,1,2,1,1,1,1,1], [1,1,1,2,2,2,2,1,1], [1]*9
            ]
        })
        # Hole 4: Island green
        defs.append({
            "layout": [
                "OOOOOOOOO",
                "O T     O",
                "O       O",
                "O  HHH  O",
                "O  HHH  O",
                "O  HHH  O",
                "O       O",
                "OOOOOOOOO",
            ],
            "height": [[1]*9 for _ in range(8)]
        })
        # Hole 5: Up a hill
        defs.append({
            "layout": [ "OOOOO", "O T O", "O   O", "O   O", "O H O", "OOOOO" ],
            "height": [ [1]*5, [1,1,1,1,1], [1,2,2,2,1], [1,3,3,3,1], [1,4,4,4,1], [1]*5 ]
        })
        # Hole 6: Down a hill
        defs.append({
            "layout": [ "OOOOO", "O T O", "O   O", "O   O", "O H O", "OOOOO" ],
            "height": [ [4]*5, [4,4,4,4,4], [4,3,3,3,4], [4,2,2,2,4], [4,1,1,1,4], [4]*5 ]
        })
        # Hole 7: Funnel
        defs.append({
            "layout": [ "OOOOOOOOO", "O T     O", "O  RRR  O", "O R H R O", "O  RRR  O", "O       O", "OOOOOOOOO" ],
            "height": [ [3]*9, [3,3,3,3,3,3,3,3,3], [3,3,2,2,2,3,3,3,3], [3,3,2,1,2,3,3,3,3], [3,3,2,2,2,3,3,3,3], [3,3,3,3,3,3,3,3,3], [3]*9 ]
        })
        # Hole 8: Ridge
        defs.append({
            "layout": [ "OOOOOOOOOOOOO", "O T         H O", "OOOOOOOOOOOOO" ],
            "height": [ [1]*13, [1,2,3,4,3,2,1,2,3,4,3,2,1], [1]*13 ]
        })
        # Hole 9: Maze
        defs.append({
            "layout": [ "OOOOOOOOO", "O T O   O", "O   O H O", "O O   O O", "O   O   O", "OOOOOOOOO" ],
            "height": [[1]*9 for _ in range(6)]
        })
        return defs

    def _setup_hole(self, hole_index):
        hole_def = self._hole_definitions[hole_index % len(self._hole_definitions)]
        self.current_hole_layout = [list(row) for row in hole_def["layout"]]
        self.current_hole_height = hole_def["height"]
        
        rows = len(self.current_hole_layout)
        cols = len(self.current_hole_layout[0])
        
        for r in range(rows):
            for c in range(cols):
                if self.current_hole_layout[r][c] == 'T':
                    self.current_tee_pos = np.array([c + 0.5, r + 0.5])
                elif self.current_hole_layout[r][c] == 'H':
                    self.current_hole_pos = np.array([c + 0.5, r + 0.5])
                    self.current_hole_layout[r][c] = ' ' # Treat it as fairway for physics

        self.ball_pos = self.current_tee_pos.copy()
        self.last_ball_pos = self.ball_pos.copy()
        self.ball_vel = np.array([0.0, 0.0])
        self.game_state = "AIMING"
        self.aim_angle = math.atan2(self.current_hole_pos[1] - self.ball_pos[1], 
                                    self.current_hole_pos[0] - self.ball_pos[0])
        self.putt_power = self.MAX_POWER / 2
        self.max_hole_dist = np.linalg.norm(self.current_tee_pos - self.current_hole_pos)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_strokes = 0
        self.current_hole_index = 0
        self.prev_space_held = False
        
        self._setup_hole(0)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, _ = action
        space_pressed = space_action == 1 and not self.prev_space_held
        self.prev_space_held = space_action == 1
        
        reward = 0
        
        if self.game_state == "AIMING":
            # Adjust aim
            if movement == 3: self.aim_angle -= self.AIM_INCREMENT
            elif movement == 4: self.aim_angle += self.AIM_INCREMENT
            # Adjust power
            if movement == 1: self.putt_power = min(self.MAX_POWER, self.putt_power + self.POWER_INCREMENT)
            elif movement == 2: self.putt_power = max(0, self.putt_power - self.POWER_INCREMENT)

            if space_pressed:
                # Putt the ball
                self.game_state = "BALL_MOVING"
                self.total_strokes += 1
                self.last_ball_pos = self.ball_pos.copy()
                
                # Sfx: putt_sound()
                
                power_ratio = self.putt_power / self.MAX_POWER
                actual_power = self.MAX_POWER * (power_ratio ** 1.5) # Make power curve non-linear for better feel
                
                self.ball_vel = np.array([
                    math.cos(self.aim_angle) * actual_power,
                    math.sin(self.aim_angle) * actual_power
                ]) / 20.0 # Scale down for physics steps
                
                reward -= 0.1 # Penalty per stroke
        
        elif self.game_state == "BALL_MOVING":
            self._update_ball_physics()

            speed = np.linalg.norm(self.ball_vel)
            if speed < self.STOP_VELOCITY_THRESHOLD:
                self.ball_vel = np.array([0.0, 0.0])
                self.game_state = "AIMING"
                
                # Sfx: ball_stop_sound()

                dist_to_hole = np.linalg.norm(self.ball_pos - self.current_hole_pos)

                if dist_to_hole < self.HOLE_RADIUS / self.TILE_WIDTH:
                    # Ball is in the hole
                    # Sfx: hole_in_sound()
                    reward += 10
                    self.current_hole_index += 1
                    if self.current_hole_index < 9:
                        self._setup_hole(self.current_hole_index)
                    else:
                        self.game_over = True
                else:
                    # Ball stopped, not in hole. Check for OOB.
                    tile_x, tile_y = int(self.ball_pos[0]), int(self.ball_pos[1])
                    rows, cols = len(self.current_hole_layout), len(self.current_hole_layout[0])
                    if not (0 <= tile_y < rows and 0 <= tile_x < cols and self.current_hole_layout[tile_y][tile_x] != 'O'):
                        # Out of Bounds
                        # Sfx: oob_splash_sound()
                        reward -= 5
                        self.total_strokes += 1 # Penalty stroke
                        self.ball_pos = self.last_ball_pos.copy()
                    else:
                        # Reward for getting closer
                        prev_dist = np.linalg.norm(self.last_ball_pos - self.current_hole_pos)
                        improvement = (prev_dist - dist_to_hole) / self.max_hole_dist
                        reward += improvement * 2.0 # Reward improvement more

        self.steps += 1
        self.score += reward
        
        terminated = self.game_over or self.total_strokes >= self.MAX_STROKES or self.steps >= self.MAX_EPISODE_STEPS
        if terminated and not self.game_over: # Game ended due to stroke/step limit
            self.game_over = True

        if terminated:
            # Calculate final score-based reward
            if self.current_hole_index >= 9 and self.total_strokes <= self.TOTAL_PAR:
                terminal_reward = 50 # Under par bonus
            elif self.total_strokes >= self.MAX_STROKES:
                terminal_reward = -50 # Penalty for maxing out strokes
            else: # Pro-rated score
                score_diff = self.total_strokes - self.TOTAL_PAR
                max_diff = self.MAX_STROKES - self.TOTAL_PAR
                # Linearly scale from +50 (at par) to -50 (at max strokes)
                terminal_reward = 50 - 100 * (score_diff / max_diff) if max_diff > 0 else 0
            
            reward += terminal_reward
            self.score += terminal_reward

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_tile_type_and_height(self, pos):
        x, y = int(pos[0]), int(pos[1])
        rows, cols = len(self.current_hole_layout), len(self.current_hole_layout[0])
        if not (0 <= y < rows and 0 <= x < cols):
            return 'O', 0
        
        height_rows, height_cols = len(self.current_hole_height), len(self.current_hole_height[0])
        height = self.current_hole_height[min(y, height_rows-1)][min(x, height_cols-1)]
        
        return self.current_hole_layout[y][x], height

    def _update_ball_physics(self):
        # Slope
        _, h_center = self._get_tile_type_and_height(self.ball_pos)
        _, h_right = self._get_tile_type_and_height(self.ball_pos + np.array([0.1, 0]))
        _, h_left = self._get_tile_type_and_height(self.ball_pos - np.array([0.1, 0]))
        _, h_down = self._get_tile_type_and_height(self.ball_pos + np.array([0, 0.1]))
        _, h_up = self._get_tile_type_and_height(self.ball_pos - np.array([0, 0.1]))

        slope_x = h_left - h_right
        slope_y = h_up - h_down
        
        slope_force = np.array([slope_x, slope_y]) * self.SLOPE_FORCE_MULTIPLIER
        self.ball_vel += slope_force

        # Friction
        self.ball_vel *= self.FRICTION
        
        # Update position
        self.ball_pos += self.ball_vel

    def _iso_to_screen(self, x, y, z=0):
        screen_x = (x - y) * self.TILE_WIDTH / 2
        screen_y = (x + y) * self.TILE_HEIGHT / 2 - (z * self.TILE_HEIGHT)
        return screen_x, screen_y

    def _update_camera(self):
        rows = len(self.current_hole_layout)
        cols = len(self.current_hole_layout[0])
        
        world_w_px, world_h_px = self._iso_to_screen(cols, rows)
        world_h_px += rows * self.TILE_HEIGHT # Adjust for height
        
        ball_screen_x, ball_screen_y = self._iso_to_screen(self.ball_pos[0], self.ball_pos[1])
        
        target_cam_x = self.width / 2 - ball_screen_x
        target_cam_y = self.height / 2 - ball_screen_y

        # Clamp camera
        min_cam_x, _ = self._iso_to_screen(cols, 0)
        max_cam_x, _ = self._iso_to_screen(0, rows)
        target_cam_x = max(min(target_cam_x, -min_cam_x + 50), -max_cam_x - self.width + 50)
        
        min_cam_y, _ = self._iso_to_screen(0,0)
        max_cam_y, _ = self._iso_to_screen(cols, rows)
        target_cam_y = max(min(target_cam_y, -min_cam_y + 50), -max_cam_y - self.height + 50)
        
        # Smooth camera movement
        self.camera_offset = self.camera_offset * 0.9 + np.array([target_cam_x, target_cam_y]) * 0.1

    def _get_observation(self):
        self._update_camera()
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x, cam_y = self.camera_offset

        # Render course tiles
        rows = len(self.current_hole_layout)
        cols = len(self.current_hole_layout[0])
        for r in range(rows):
            for c in range(cols):
                tile_type, height = self._get_tile_type_and_height(np.array([c, r]))
                
                if tile_type == ' ': color = self.COLOR_GREEN[height % len(self.COLOR_GREEN)]
                elif tile_type == 'R': color = self.COLOR_ROUGH
                else: color = self.COLOR_OOB
                
                p1 = self._iso_to_screen(c, r, height)
                p2 = self._iso_to_screen(c + 1, r, height)
                p3 = self._iso_to_screen(c + 1, r + 1, height)
                p4 = self._iso_to_screen(c, r + 1, height)

                points = [(p[0] + cam_x, p[1] + cam_y) for p in [p1, p2, p3, p4]]
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Render hole
        _, hole_h = self._get_tile_type_and_height(self.current_hole_pos)
        hole_scr_x, hole_scr_y = self._iso_to_screen(self.current_hole_pos[0], self.current_hole_pos[1], hole_h)
        pygame.gfxdraw.filled_circle(self.screen, int(hole_scr_x + cam_x), int(hole_scr_y + cam_y), self.HOLE_RADIUS, self.COLOR_HOLE)
        
        # Render flag
        flag_pole_top = (int(hole_scr_x + cam_x), int(hole_scr_y + cam_y - 40))
        flag_pole_bot = (int(hole_scr_x + cam_x), int(hole_scr_y + cam_y))
        pygame.draw.line(self.screen, self.COLOR_FLAG, flag_pole_top, flag_pole_bot, 1)
        flag_points = [flag_pole_top, (flag_pole_top[0]-15, flag_pole_top[1]+5), (flag_pole_top[0], flag_pole_top[1]+10)]
        pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)

        # Render ball
        _, ball_h = self._get_tile_type_and_height(self.ball_pos)
        ball_scr_x, ball_scr_y = self._iso_to_screen(self.ball_pos[0], self.ball_pos[1], ball_h)
        
        # Shadow
        shadow_surf = pygame.Surface((self.BALL_RADIUS*2, self.BALL_RADIUS*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_ellipse(shadow_surf, self.BALL_RADIUS, self.BALL_RADIUS, self.BALL_RADIUS, self.BALL_RADIUS//2, self.COLOR_SHADOW)
        self.screen.blit(shadow_surf, (int(ball_scr_x + cam_x - self.BALL_RADIUS), int(ball_scr_y + cam_y - self.BALL_RADIUS//2)))
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(ball_scr_x + cam_x), int(ball_scr_y + cam_y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(ball_scr_x + cam_x), int(ball_scr_y + cam_y), self.BALL_RADIUS, self.COLOR_BALL)

        # Render aim guide
        if self.game_state == "AIMING":
            line_len = 20 + (self.putt_power / self.MAX_POWER) * 80
            end_x = ball_scr_x + math.cos(self.aim_angle) * line_len
            end_y = ball_scr_y - self.BALL_RADIUS + math.sin(self.aim_angle) * line_len / 2 # Isometric projection for line
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, (ball_scr_x + cam_x, ball_scr_y + cam_y - self.BALL_RADIUS), (end_x + cam_x, end_y + cam_y))

    def _render_ui(self):
        # Hole info
        hole_text = self.font_small.render(f"HOLE {self.current_hole_index + 1}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hole_text, (self.width - hole_text.get_width() - 10, 10))

        # Stroke info
        stroke_text = self.font_small.render(f"STROKES: {self.total_strokes}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stroke_text, (10, 10))
        
        par_text = self.font_small.render(f"PAR: {self.TOTAL_PAR}", True, self.COLOR_UI_TEXT)
        self.screen.blit(par_text, (10, 30))
        
        # Power bar
        if self.game_state == "AIMING":
            bar_w, bar_h = 150, 20
            bar_x, bar_y = (self.width - bar_w) / 2, self.height - bar_h - 10
            power_ratio = self.putt_power / self.MAX_POWER
            fill_w = bar_w * power_ratio
            
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            if fill_w > 0:
                pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.current_hole_index >= 9:
                msg = "COURSE COMPLETE!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.width / 2, self.height / 2 - 20))
            
            final_score_text = self.font_small.render(f"Final Strokes: {self.total_strokes} (Par: {self.TOTAL_PAR})", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.width / 2, self.height / 2 + 30))
            
            overlay.blit(end_text, end_rect)
            overlay.blit(final_score_text, final_score_rect)
            self.screen.blit(overlay, (0, 0))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "total_strokes": self.total_strokes,
            "current_hole": self.current_hole_index + 1,
            "game_state": self.game_state,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    # Game loop
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # For a proper display window, we need to create one
        if 'display' not in locals():
            display = pygame.display.set_mode((env.width, env.height))
            pygame.display.set_caption("Isometric Golf")
        
        display.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS

    # Keep the final screen visible for a moment
    if info.get("total_strokes"):
        print(f"Game Over! Final Strokes: {info['total_strokes']}")
    
    pygame.time.wait(2000)
    env.close()