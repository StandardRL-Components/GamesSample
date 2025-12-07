
# Generated: 2025-08-28T06:40:20.520535
# Source Brief: brief_03001.md
# Brief Index: 3001

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to change power, ←/→ to aim. Space to shoot. Shift to reset aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Master challenging isometric mini-golf courses. Complete all 9 holes in under 30 strokes to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STROKES = 30
        self.NUM_HOLES = 9
        self.MAX_EPISODE_STEPS = 1000 # Safety break

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GREEN = (64, 135, 84)
        self.COLOR_ROUGH = (48, 107, 66)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_OBSTACLE_BORDER = (80, 90, 100)
        self.COLOR_HOLE = (20, 20, 20)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_AIM_LINE = (255, 255, 255, 150)
        self.COLOR_POWER_BAR_BG = (50, 50, 50)

        # Physics constants
        self.FRICTION = 0.98
        self.MIN_VELOCITY = 0.02
        self.MAX_SHOT_SPEED = 8.0
        self.BALL_RADIUS_WORLD = 0.4
        self.HOLE_RADIUS_WORLD = 0.6
        self.STUCK_THRESHOLD = 15 # steps

        # Isometric projection constants
        self.ISO_TILE_WIDTH = 32
        self.ISO_TILE_HEIGHT = 16
        self.ISO_ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ISO_ORIGIN_Y = 60

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Hole definitions
        self._define_holes()

        # Initialize state variables
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_is_moving = False
        self.aim_angle = 0.0
        self.shot_power = 0.0
        self.total_strokes = 0
        self.current_hole_index = 0
        self.current_hole_data = {}
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.stuck_counter = 0
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def _define_holes(self):
        self.hole_definitions = []
        # Hole 1: Straight shot
        self.hole_definitions.append({
            "start_pos": np.array([15.0, 5.0]), "hole_pos": np.array([15.0, 15.0]),
            "bounds": pygame.Rect(10, 2, 10, 16), "obstacles": []
        })
        # Hole 2: Simple obstacle
        self.hole_definitions.append({
            "start_pos": np.array([5.0, 10.0]), "hole_pos": np.array([25.0, 10.0]),
            "bounds": pygame.Rect(2, 5, 26, 10),
            "obstacles": [pygame.Rect(14, 9, 2, 2)]
        })
        # Hole 3: L-shape
        self.hole_definitions.append({
            "start_pos": np.array([5.0, 5.0]), "hole_pos": np.array([18.0, 18.0]),
            "bounds": pygame.Rect(3, 3, 17, 17),
            "obstacles": [pygame.Rect(3, 10, 10, 10), pygame.Rect(10, 3, 10, 10)]
        })
        # Hole 4: Corridor
        self.hole_definitions.append({
            "start_pos": np.array([4.0, 10.0]), "hole_pos": np.array([26.0, 10.0]),
            "bounds": pygame.Rect(2, 8, 26, 4),
            "obstacles": [pygame.Rect(10, 8, 2, 1), pygame.Rect(10, 11, 2, 1), pygame.Rect(18, 8, 2, 1), pygame.Rect(18, 11, 2, 1)]
        })
        # Hole 5: Ricochet
        self.hole_definitions.append({
            "start_pos": np.array([5.0, 5.0]), "hole_pos": np.array([20.0, 5.0]),
            "bounds": pygame.Rect(3, 3, 20, 15),
            "obstacles": [pygame.Rect(12, 3, 2, 10)]
        })
        # Hole 6: Funnel
        self.hole_definitions.append({
            "start_pos": np.array([15.0, 4.0]), "hole_pos": np.array([15.0, 18.0]),
            "bounds": pygame.Rect(5, 2, 20, 18),
            "obstacles": [pygame.Rect(5, 12, 8, 2), pygame.Rect(17, 12, 8, 2)]
        })
        # Hole 7: Maze
        self.hole_definitions.append({
            "start_pos": np.array([4.0, 4.0]), "hole_pos": np.array([20.0, 16.0]),
            "bounds": pygame.Rect(2, 2, 20, 16),
            "obstacles": [pygame.Rect(2, 8, 12, 2), pygame.Rect(8, 12, 14, 2)]
        })
        # Hole 8: Island
        self.hole_definitions.append({
            "start_pos": np.array([15.0, 5.0]), "hole_pos": np.array([15.0, 15.0]),
            "bounds": pygame.Rect(12, 12, 6, 6),
            "obstacles": [] # The bounds are the obstacle
        })
        # Hole 9: The Gauntlet
        self.hole_definitions.append({
            "start_pos": np.array([3.0, 10.0]), "hole_pos": np.array([28.0, 10.0]),
            "bounds": pygame.Rect(1, 9, 29, 2.5),
            "obstacles": [pygame.Rect(8, 9, 1, 1), pygame.Rect(13, 10.5, 1, 1), pygame.Rect(18, 9, 1, 1), pygame.Rect(23, 10.5, 1, 1)]
        })

    def _setup_hole(self, hole_index):
        self.current_hole_index = hole_index
        self.current_hole_data = self.hole_definitions[hole_index]
        self.ball_pos = self.current_hole_data["start_pos"].copy()
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_is_moving = False
        self.aim_angle = -math.pi / 2  # Straight up in world coords
        self.shot_power = 0.0
        self.stuck_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.total_strokes = 0
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.particles = []
        
        self._setup_hole(0)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()
        
        if self.ball_is_moving:
            reward += self._update_ball_physics()
        else:
            reward += self._handle_player_input(action)
        
        self._update_particles()
        
        if self.total_strokes >= self.MAX_STROKES and not self.game_won:
            self.game_over = True
            terminated = True
            reward -= 100
        
        if self.game_won:
            terminated = True
            # Scaled reward for finishing under par
            reward += 50 + (self.MAX_STROKES - self.total_strokes)

        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Aiming and Power
        if movement == 1: self.shot_power = min(1.0, self.shot_power + 0.05) # Up -> Power Up
        if movement == 2: self.shot_power = max(0.0, self.shot_power - 0.05) # Down -> Power Down
        if movement == 3: self.aim_angle -= 0.08 # Left -> Aim Left
        if movement == 4: self.aim_angle += 0.08 # Right -> Aim Right
        if shift_press: self.aim_angle = -math.pi / 2 # Reset aim

        # Normalize angle
        self.aim_angle = self.aim_angle % (2 * math.pi)

        # Shoot
        if space_press and self.shot_power > 0.05:
            # sfx: golf_swing.wav
            self.total_strokes += 1
            reward -= 1.0
            self.ball_is_moving = True
            speed = self.shot_power * self.MAX_SHOT_SPEED
            self.ball_vel = np.array([speed * math.cos(self.aim_angle), speed * math.sin(self.aim_angle)])
            self.shot_power = 0.0 # Reset power after shot

        return reward

    def _update_ball_physics(self):
        reward = -0.1 # Small penalty for each step the ball is moving
        
        self.ball_pos += self.ball_vel
        self.ball_vel *= self.FRICTION

        # Collision with bounds
        bounds = self.current_hole_data["bounds"]
        if self.ball_pos[0] < bounds.left + self.BALL_RADIUS_WORLD or self.ball_pos[0] > bounds.right - self.BALL_RADIUS_WORLD:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], bounds.left + self.BALL_RADIUS_WORLD, bounds.right - self.BALL_RADIUS_WORLD)
            reward -= 0.5; self._create_particles(self.ball_pos, 5) # sfx: bounce.wav
        if self.ball_pos[1] < bounds.top + self.BALL_RADIUS_WORLD or self.ball_pos[1] > bounds.bottom - self.BALL_RADIUS_WORLD:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], bounds.top + self.BALL_RADIUS_WORLD, bounds.bottom - self.BALL_RADIUS_WORLD)
            reward -= 0.5; self._create_particles(self.ball_pos, 5) # sfx: bounce.wav

        # Collision with obstacles
        for obs in self.current_hole_data["obstacles"]:
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS_WORLD, self.ball_pos[1] - self.BALL_RADIUS_WORLD, self.BALL_RADIUS_WORLD*2, self.BALL_RADIUS_WORLD*2)
            if obs.colliderect(ball_rect):
                # A simple reflection logic
                dx = self.ball_pos[0] - obs.centerx
                dy = self.ball_pos[1] - obs.centery
                if abs(dx) > abs(dy): self.ball_vel[0] *= -1
                else: self.ball_vel[1] *= -1
                reward -= 0.5; self._create_particles(self.ball_pos, 5) # sfx: bounce_hard.wav
                # Push ball out of obstacle to prevent sticking
                while obs.colliderect(pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS_WORLD, self.ball_pos[1] - self.BALL_RADIUS_WORLD, self.BALL_RADIUS_WORLD*2, self.BALL_RADIUS_WORLD*2)):
                    self.ball_pos += self.ball_vel * 0.1
                break
        
        # Check if ball is in hole
        hole_pos = self.current_hole_data["hole_pos"]
        dist_to_hole = np.linalg.norm(self.ball_pos - hole_pos)
        vel_mag = np.linalg.norm(self.ball_vel)
        if dist_to_hole < self.HOLE_RADIUS_WORLD and vel_mag < self.MAX_SHOT_SPEED / 2.5:
            # sfx: hole_sink.wav
            reward += 5.0
            self._create_particles(hole_pos, 20, (255, 215, 0))
            self.current_hole_index += 1
            if self.current_hole_index >= self.NUM_HOLES:
                self.game_won = True
            else:
                self._setup_hole(self.current_hole_index)
            return reward

        # Check if ball has stopped
        if vel_mag < self.MIN_VELOCITY:
            self.ball_vel = np.array([0.0, 0.0])
            self.ball_is_moving = False
            self.stuck_counter = 0
        elif vel_mag < self.MIN_VELOCITY * 5:
            self.stuck_counter += 1
        
        # Anti-softlock mechanism
        if self.stuck_counter > self.STUCK_THRESHOLD:
            self.total_strokes += 1 # Penalty stroke
            reward -= 1.0
            self._setup_hole(self.current_hole_index) # Reset to hole start
            # sfx: reset.wav
            
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _world_to_iso(self, x, y):
        iso_x = self.ISO_ORIGIN_X + (x - y) * self.ISO_TILE_WIDTH / 2
        iso_y = self.ISO_ORIGIN_Y + (x + y) * self.ISO_TILE_HEIGHT / 2
        return int(iso_x), int(iso_y)

    def _render_game(self):
        # Draw course bounds
        bounds = self.current_hole_data["bounds"]
        p1 = self._world_to_iso(bounds.left, bounds.top)
        p2 = self._world_to_iso(bounds.right, bounds.top)
        p3 = self._world_to_iso(bounds.right, bounds.bottom)
        p4 = self._world_to_iso(bounds.left, bounds.bottom)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), self.COLOR_ROUGH)
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), self.COLOR_GREEN)

        # Draw obstacles
        for obs in self.current_hole_data["obstacles"]:
            o1 = self._world_to_iso(obs.left, obs.top)
            o2 = self._world_to_iso(obs.right, obs.top)
            o3 = self._world_to_iso(obs.right, obs.bottom)
            o4 = self._world_to_iso(obs.left, obs.bottom)
            pygame.gfxdraw.filled_polygon(self.screen, (o1, o2, o3, o4), self.COLOR_OBSTACLE)
            pygame.gfxdraw.aapolygon(self.screen, (o1, o2, o3, o4), self.COLOR_OBSTACLE_BORDER)

        # Draw hole
        hole_pos_iso = self._world_to_iso(*self.current_hole_data["hole_pos"])
        hole_radius_iso = int(self.HOLE_RADIUS_WORLD * self.ISO_TILE_WIDTH / 2)
        pygame.gfxdraw.filled_circle(self.screen, hole_pos_iso[0], hole_pos_iso[1], hole_radius_iso, self.COLOR_HOLE)
        
        # Draw particles
        for p in self.particles:
            p_iso_x, p_iso_y = self._world_to_iso(p['pos'][0], p['pos'][1])
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, p_iso_x, p_iso_y, int(p['radius']), color)

        # Draw aiming line
        if not self.ball_is_moving and not self.game_over and not self.game_won:
            ball_pos_iso = self._world_to_iso(*self.ball_pos)
            aim_length = 5 + self.shot_power * 70
            end_x = ball_pos_iso[0] + aim_length * math.cos(self.aim_angle - math.pi/2)
            end_y = ball_pos_iso[1] + aim_length * math.sin(self.aim_angle - math.pi/2)
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, ball_pos_iso, (end_x, end_y), 2)

        # Draw ball
        ball_pos_iso = self._world_to_iso(*self.ball_pos)
        ball_radius_iso = int(self.BALL_RADIUS_WORLD * self.ISO_TILE_WIDTH / 2)
        # Simple shadow
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_iso[0], ball_pos_iso[1] + 3, ball_radius_iso, (0,0,0,50))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_iso[0], ball_pos_iso[1], ball_radius_iso, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_iso[0], ball_pos_iso[1], ball_radius_iso, (200,200,200))

    def _render_ui(self):
        # Text
        hole_text = self.font_main.render(f"Hole: {self.current_hole_index + 1}/{self.NUM_HOLES}", True, self.COLOR_UI_TEXT)
        strokes_text = self.font_main.render(f"Strokes: {self.total_strokes}/{self.MAX_STROKES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hole_text, (10, 10))
        self.screen.blit(strokes_text, (10, 35))

        # Power bar
        if not self.ball_is_moving and not self.game_over and not self.game_won:
            bar_x, bar_y, bar_w, bar_h = 10, self.SCREEN_HEIGHT - 30, 150, 20
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            power_w = self.shot_power * bar_w
            power_color = (int(255 * self.shot_power), int(255 * (1 - self.shot_power)), 0)
            pygame.draw.rect(self.screen, power_color, (bar_x, bar_y, power_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

        # Game Over / Win Text
        if self.game_over:
            text_surf = self.font_large.render("GAME OVER", True, (200, 50, 50))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)
        elif self.game_won:
            text_surf = self.font_large.render("YOU WIN!", True, (50, 200, 50))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "total_strokes": self.total_strokes,
            "current_hole": self.current_hole_index + 1,
            "steps": self.steps,
        }
        
    def _create_particles(self, pos, count, color=(200, 200, 200)):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': random.randint(10, 20),
                'max_life': 20,
                'radius': random.uniform(1, 4),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel'] * 0.1
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Mini-Golf")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_press = 1 if keys[pygame.K_SPACE] else 0
        shift_press = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_press, shift_press]
        # --- End action mapping ---

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            # Optional: auto-reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we control the step rate.
        # For a responsive human experience, we run this loop quickly.
        # The game logic in step() only progresses on a "shoot" action.
        clock.tick(30)

    env.close()