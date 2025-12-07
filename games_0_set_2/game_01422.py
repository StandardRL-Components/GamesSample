
# Generated: 2025-08-27T17:05:29.722358
# Source Brief: brief_01422.md
# Brief Index: 1422

        
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

    user_guide = (
        "Controls: ←→ to aim, hold [space] to charge power, release to shoot. Hold [shift] to reduce power."
    )

    game_description = (
        "A procedurally generated isometric mini-golf game. Aim and shoot to sink the ball in all 9 holes with the fewest strokes possible."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.NUM_HOLES = 9
        self.MAX_HOLE_STROKES = 6
        self.MAX_TOTAL_STROKES = self.NUM_HOLES * self.MAX_HOLE_STROKES
        self.PAR_TOTAL = 3 * self.NUM_HOLES # Par 3 for each hole

        # --- Physics Constants ---
        self.MAX_POWER = 1.0
        self.MAX_SPEED = 15.0
        self.FRICTION = 0.98
        self.STOP_THRESHOLD = 0.1
        self.BOUNCE_FACTOR = 0.8
        self.BALL_RADIUS = 8
        self.HOLE_RADIUS = 12

        # --- Visual Constants ---
        self.COLOR_BG = (20, 80, 40)
        self.COLOR_WALL = (240, 240, 240)
        self.COLOR_WALL_SHADOW = (180, 180, 180)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_SHADOW = (200, 200, 0)
        self.COLOR_HOLE = (255, 0, 0)
        self.COLOR_HOLE_INNER = (0, 0, 0)
        self.COLOR_AIM = (0, 192, 255, 150)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_POWER_BAR_BG = (50, 50, 50)
        self.COLOR_POWER_BAR_FILL = (255, 100, 0)
        
        # --- Isometric Projection ---
        self.ISO_TILE_WIDTH_HALF = 16
        self.ISO_TILE_HEIGHT_HALF = 8
        self.WORLD_OFFSET = [self.WIDTH // 2, 80]
        self.WORLD_WIDTH = 30
        self.WORLD_HEIGHT = 20

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_huge = pygame.font.Font(None, 72)

        # --- State Variables ---
        self.total_strokes = 0
        self.current_hole_num = 0
        self.game_over = False
        self.win = False
        self.hole_strokes = 0
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_in_motion = False
        self.aim_angle_rad = 0.0
        self.shot_power = 0.0
        self.hole_pos = np.array([0.0, 0.0])
        self.walls = []
        self.particles = []
        self.last_dist_to_hole = float('inf')

        self.reset()
        
        # This check is for development and ensures the implementation conforms to the API
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_strokes = 0
        self.current_hole_num = 1
        self.game_over = False
        self.win = False
        self.particles = []

        self._setup_new_hole()

        return self._get_observation(), self._get_info()

    def _setup_new_hole(self):
        self.hole_strokes = 0
        self.ball_pos = np.array([5.0, self.WORLD_HEIGHT / 2])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_in_motion = False
        self.aim_angle_rad = 0.0
        self.shot_power = 0.0
        
        self.hole_pos = np.array([self.WORLD_WIDTH - 5.0, self.WORLD_HEIGHT / 2])
        self.last_dist_to_hole = np.linalg.norm(self.ball_pos - self.hole_pos)

        # Procedural Generation
        self.walls = []
        num_walls = min(5, (self.current_hole_num -1) // 2)
        
        for _ in range(num_walls):
            is_vertical = self.np_random.choice([True, False])
            if is_vertical:
                w_x = self.np_random.uniform(self.ball_pos[0] + 5, self.hole_pos[0] - 5)
                w_y = self.np_random.uniform(2, self.WORLD_HEIGHT - 8)
                w_w = 0.5
                w_h = self.np_random.uniform(4, 8)
            else:
                w_x = self.np_random.uniform(self.ball_pos[0] + 3, self.hole_pos[0] - 8)
                w_y = self.np_random.uniform(2, self.WORLD_HEIGHT - 3)
                w_w = self.np_random.uniform(4, 8)
                w_h = 0.5
            
            self.walls.append(pygame.Rect(w_x, w_y, w_w, w_h))


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small cost for taking a step (thinking)
        terminated = False
        shot_triggered = False

        # --- AIMING PHASE ---
        if not self.ball_in_motion:
            if movement == 3:  # Left
                self.aim_angle_rad += 0.08
            if movement == 4:  # Right
                self.aim_angle_rad -= 0.08

            if space_held:
                self.shot_power = min(self.MAX_POWER, self.shot_power + 0.05)
            if shift_held:
                self.shot_power = max(0.0, self.shot_power - 0.05)
            
            if not space_held and self.shot_power > 0:
                shot_triggered = True

        # --- SHOT & SIMULATION PHASE ---
        if shot_triggered:
            # sfx: club_swing
            self.ball_in_motion = True
            self.hole_strokes += 1
            self.total_strokes += 1
            
            power_scalar = self.shot_power * self.MAX_SPEED
            self.ball_vel = np.array([
                math.cos(self.aim_angle_rad) * power_scalar,
                -math.sin(self.aim_angle_rad) * power_scalar # Y is inverted in world space
            ])
            self.shot_power = 0.0
            
            dist_before_shot = np.linalg.norm(self.ball_pos - self.hole_pos)
            sunk_ball = False
            wall_hit_penalty = 0

            # Physics simulation loop
            for _ in range(500): # Max simulation steps
                if np.linalg.norm(self.ball_vel) < self.STOP_THRESHOLD:
                    break

                prev_pos = self.ball_pos.copy()
                self.ball_pos += self.ball_vel / 30.0 # Scale velocity by assumed FPS
                self.ball_vel *= self.FRICTION
                
                # Wall collisions
                for wall in self.walls:
                    if wall.collidepoint(self.ball_pos):
                        # sfx: ball_hit_wall
                        wall_hit_penalty += -5
                        self._create_particles(self.ball_pos, 5, (200,200,200))

                        # Simple penetration response
                        if prev_pos[0] < wall.left or prev_pos[0] > wall.right:
                            self.ball_vel[0] *= -self.BOUNCE_FACTOR
                        if prev_pos[1] < wall.top or prev_pos[1] > wall.bottom:
                            self.ball_vel[1] *= -self.BOUNCE_FACTOR
                        self.ball_pos = prev_pos # Revert position to avoid getting stuck
                        break

                # Boundary collisions
                if not (0 < self.ball_pos[0] < self.WORLD_WIDTH and 0 < self.ball_pos[1] < self.WORLD_HEIGHT):
                    if not (0 < self.ball_pos[0] < self.WORLD_WIDTH): self.ball_vel[0] *= -1
                    if not (0 < self.ball_pos[1] < self.WORLD_HEIGHT): self.ball_vel[1] *= -1
                    self.ball_pos = np.clip(self.ball_pos, [0,0], [self.WORLD_WIDTH, self.WORLD_HEIGHT])

                # Check for sinking the ball
                if np.linalg.norm(self.ball_pos - self.hole_pos) < self.HOLE_RADIUS / self.ISO_TILE_WIDTH_HALF:
                    # sfx: ball_in_hole
                    sunk_ball = True
                    self._create_particles(self.hole_pos, 20, self.COLOR_BALL)
                    break

            self.ball_in_motion = False
            self.ball_vel = np.array([0.0, 0.0])
            dist_after_shot = np.linalg.norm(self.ball_pos - self.hole_pos)

            # Calculate reward for the shot
            reward = wall_hit_penalty
            reward += (dist_before_shot - dist_after_shot) # Distance reduction reward

            if sunk_ball:
                reward += 10
                self.current_hole_num += 1
                if self.current_hole_num > self.NUM_HOLES:
                    self.win = True
                    self.game_over = True
                    terminated = True
                    if self.total_strokes <= self.PAR_TOTAL:
                        reward += 100
                else:
                    self._setup_new_hole()
            else:
                self.last_dist_to_hole = dist_after_shot
                if self.hole_strokes >= self.MAX_HOLE_STROKES:
                    self.game_over = True
                    terminated = True
                    reward -= 5

        if self.total_strokes >= self.MAX_TOTAL_STROKES:
            self.game_over = True
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _to_iso(self, pos):
        world_x, world_y = pos
        iso_x = self.WORLD_OFFSET[0] + (world_x - world_y) * self.ISO_TILE_WIDTH_HALF
        iso_y = self.WORLD_OFFSET[1] + (world_x + world_y) * self.ISO_TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw course outline (for visual reference)
        course_corners = [
            self._to_iso((0, 0)),
            self._to_iso((self.WORLD_WIDTH, 0)),
            self._to_iso((self.WORLD_WIDTH, self.WORLD_HEIGHT)),
            self._to_iso((0, self.WORLD_HEIGHT))
        ]
        pygame.draw.polygon(self.screen, (25, 95, 45), course_corners)
        pygame.draw.aalines(self.screen, (30, 110, 55), True, course_corners)

        # Draw hole
        hole_iso = self._to_iso(self.hole_pos)
        pygame.gfxdraw.filled_circle(self.screen, hole_iso[0], hole_iso[1], self.HOLE_RADIUS, self.COLOR_HOLE_INNER)
        pygame.gfxdraw.aacircle(self.screen, hole_iso[0], hole_iso[1], self.HOLE_RADIUS, self.COLOR_HOLE)

        # Draw walls
        for wall in self.walls:
            p1 = self._to_iso(wall.topleft)
            p2 = self._to_iso(wall.topright)
            p3 = self._to_iso(wall.bottomright)
            p4 = self._to_iso(wall.bottomleft)
            pygame.draw.polygon(self.screen, self.COLOR_WALL_SHADOW, [p1, p2, p3, p4])
            pygame.draw.aalines(self.screen, self.COLOR_WALL, True, [p1, p2, p3, p4])

        # Draw particles
        self._update_and_draw_particles()

        # Draw ball
        ball_iso = self._to_iso(self.ball_pos)
        pygame.gfxdraw.filled_circle(self.screen, ball_iso[0], ball_iso[1] - 3, self.BALL_RADIUS, self.COLOR_BALL_SHADOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_iso[0], ball_iso[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_iso[0], ball_iso[1], self.BALL_RADIUS, self.COLOR_BALL_SHADOW)

        # Draw aim line and power bar if not in motion
        if not self.ball_in_motion:
            # Aim line
            aim_len = 50 + self.shot_power * 100
            end_x = ball_iso[0] + aim_len * math.cos(self.aim_angle_rad)
            end_y = ball_iso[1] - aim_len * math.sin(self.aim_angle_rad) # Screen Y is inverted
            pygame.draw.aaline(self.screen, self.COLOR_AIM, ball_iso, (end_x, end_y))

            # Power bar
            bar_x, bar_y, bar_w, bar_h = 20, self.HEIGHT - 40, 200, 20
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            fill_w = self.shot_power * bar_w
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

    def _render_ui(self):
        hole_text = self.font_large.render(f"Hole {self.current_hole_num}/{self.NUM_HOLES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hole_text, (20, 20))

        strokes_text = self.font_small.render(f"Strokes: {self.hole_strokes}", True, self.COLOR_UI_TEXT)
        self.screen.blit(strokes_text, (20, 60))

        total_text = self.font_small.render(f"Total: {self.total_strokes}", True, self.COLOR_UI_TEXT)
        self.screen.blit(total_text, (20, 80))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_huge.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)

            score_text = self.font_large.render(f"Total Strokes: {self.total_strokes}", True, self.COLOR_UI_TEXT)
            score_rect = score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 40))
            self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": -self.total_strokes, # Higher score is better, so invert strokes
            "steps": self.total_strokes,
            "hole": self.current_hole_num,
            "hole_strokes": self.hole_strokes,
        }
    
    def _create_particles(self, pos, count, color):
        pos_iso = self._to_iso(pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos_iso),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p["life"] * 15)))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color, (0,0,4,4))
                self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])))

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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    terminated = False
    
    # Use a window to display the game
    pygame.display.set_caption("Isometric Mini-Golf")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Map keys to MultiDiscrete action space
        if keys[pygame.K_UP]:
            pass # Not used in this mapping
        if keys[pygame.K_DOWN]:
            pass # Not used
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # The game is turn-based, so we can slow down the loop for human play
        env.clock.tick(30)

    env.close()